'''
Copyright (C) 2023 TuringVision

Detectron2 custom trainer.
'''
import os.path as osp
import logging
from detectron2.engine import DefaultTrainer
from detectron2.utils.events import (EventWriter, JSONWriter,
                                     TensorboardXWriter,
                                     get_event_storage)
from detectron2.modeling import build_model
from fastprogress.fastprogress import master_bar, progress_bar


class CustomMetricPrinter(EventWriter):
    """
    Print __common__ metrics to the terminal, including
    iteration memory, all losses, and the learning rate.
    """

    def __init__(self, max_iter, iters_per_epoch=None, callback=None):
        self._max_iter = max_iter
        self._cb = callback
        self.t_iter = master_bar(range(max_iter))
        self.t_iter.update(0)
        self._title_inited = False

        if callback:
            self._epoch = 0
            self._epochs = int(max_iter / iters_per_epoch)
            self._iters_per_epoch = iters_per_epoch
        self.show()

    def show(self):
        if hasattr(self.t_iter, 'show'):
            self.t_iter.show()

    def write(self):
        import torch
        storage = get_event_storage()
        iteration = storage.iter + 1

        if iteration > self._max_iter:
            return

        try:
            lr = "{:.6f}".format(storage.history("lr").latest())
        except KeyError:
            lr = "N/A"

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None
        memory = "{:.0f}M".format(max_mem_mb) if max_mem_mb is not None else ""

        loss_items = {k: v for k, v in storage.histories().items()
                      if 'loss' in k}
        acc_items = {k: v for k, v in storage.histories().items()
                     if 'acc' in k}

        if not self._title_inited:
            items = ['iter'] + list(loss_items.keys()) + list(acc_items.keys())
            items += ['lr', 'max_mem']
            self.t_iter.names = list(loss_items.keys())
            self.t_iter.write(items, table=True)
            self._title_inited = True

        values = [str(iteration)]
        values += ['{:.3f}'.format(v.median(20))
                   for v in loss_items.values()]
        values += ['{:.3f}'.format(v.median(20))
                   for v in acc_items.values()]
        values += [lr, memory]

        graphs = [[[d[1] for d in v.values()], [d[0] for d in v.values()]]
                  for v in loss_items.values()]

        if self._cb:
            epoch_current = int(iteration / self._iters_per_epoch)
            percent_in_epoch = int(
                (iteration % self._iters_per_epoch) / self._iters_per_epoch * 100)
            status = {
                'desc': 'training_epoch%d' % (epoch_current if epoch_current > self._epoch else self._epoch),
                'percent': percent_in_epoch,
            }
            self._cb(status)

            if epoch_current > self._epoch:
                fg_cls_keys = [
                    k for k in acc_items.keys() if 'fg_cls_accuracy' in k]
                fg_cls_acc = round(acc_items[fg_cls_keys[0]].median(
                    20), 6) if fg_cls_keys else None
                total_loss = round(loss_items['total_loss'].median(20), 6) \
                    if 'total_loss' in loss_items.keys() else None

                status = {
                    'desc': 'training_epoch%d_iter%d' % (self._epoch, iteration),
                    'percent': int(100*iteration/self._max_iter),
                    'epoch': self._epoch,
                    'epochs': self._epochs,
                    'pass_desc': 'epoch_%d' % (self._epochs),
                    'acc': fg_cls_acc,
                    'val_acc': 0.0,
                    'loss': total_loss,
                    'val_loss': 0.0,
                    'fix_layer': 'group',
                }

                addition_dict = {'iter': iteration,
                                 'lr': float(lr), 'max_mem': memory}
                for k, v in storage.histories().items():
                    if 'loss' in k or 'acc' in k:
                        addition_dict[k] = round(v.median(20), 6)
                status['addition'] = addition_dict
                self._cb(status)
                self._epoch = epoch_current

        self.t_iter.update_graph(graphs)
        self.t_iter.write(values, table=True)
        self.t_iter.update(iteration)
        self.show()

    def close(self):
        self.t_iter.on_iter_end()
        self.show()


class Detectron2Trainer(DefaultTrainer):
    """
    A trainer with default training logic.
    """

    def __init__(self, cfg, train_data_loader, valid_data_loader, callback=None):
        """
        Args:
            cfg (CfgNode):
            train_data_loader:
            valid_data_loader:
        """
        self.train_data_loader = train_data_loader
        self.valid_data_loader = valid_data_loader
        self._cb = callback
        super().__init__(cfg)
        logger = logging.getLogger("detectron2")
        logger.setLevel(logging.WARNING)

    def build_train_loader(self, cfg=None):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_train_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return self.train_data_loader

    @classmethod
    def build_model(cls, cfg):
        """
        Returns:
            torch.nn.Module:

        It now calls :func:`detectron2.modeling.build_model`.
        Overwrite it if you'd like a different model.
        """
        model = build_model(cfg)
        return model

    def build_test_loader(self, cfg=None, dataset_name=None):
        """
        Returns:
            iterable

        It now calls :func:`detectron2.data.build_detection_test_loader`.
        Overwrite it if you'd like a different data loader.
        """
        return self.valid_data_loader

    def build_writers(self):
        """
        Build a list of writers to be used. By default it contains
        writers that write metrics to the screen,
        a json file, and a tensorboard event file respectively.
        If you'd like a different list of writers, you can overwrite it in
        your trainer.

        Returns:
            list[EventWriter]: a list of :class:`EventWriter` objects.
        """
        if self._cb:
            data_length = len(self.train_data_loader.dataset.dataset)
            batch_size = self.train_data_loader.batch_size
            iters_per_epoch_f = data_length / batch_size
        else:
            iters_per_epoch_f = None

        # Assume the default print/log frequency.
        return [
            # It may not always print what you want to see, since it prints "common" metrics only.
            CustomMetricPrinter(self.max_iter, iters_per_epoch_f, self._cb),
            JSONWriter(osp.join(self.cfg.OUTPUT_DIR, "metrics.json")),
            TensorboardXWriter(self.cfg.OUTPUT_DIR),
        ]

    @classmethod
    def test(cls, cfg, model, evaluators=None):
        """
        Args:
            cfg (CfgNode):
            model (nn.Module):
            evaluators (list[DatasetEvaluator] or None): if None, will call
                :meth:`build_evaluator`. Otherwise, must have the same length as
                `cfg.DATASETS.TEST`.

        Returns:
            dict: a dict of result metrics
        """
        return None
