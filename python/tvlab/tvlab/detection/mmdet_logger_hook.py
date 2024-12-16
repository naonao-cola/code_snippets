'''
Copyright (C) 2023 TuringVision

logger hook for mmdetection.
'''
from mmcv.runner import HOOKS, LoggerHook
from fastprogress.fastprogress import master_bar


@HOOKS.register_module()
class MMdetLoggerHook(LoggerHook):
    def __init__(self, by_epoch=False, interval=10, ignore_last=True, reset_flag=False):
        super(MMdetLoggerHook, self).__init__(interval, ignore_last, reset_flag, by_epoch=by_epoch)
        self.t_iter = None
        self.last_iter = 0
        self.max_iter = 0
        self._title_inited = False

    def before_run(self, runner):
        self.t_iter = master_bar(range(runner.max_iters))
        self.t_iter.update(0)
        self.show()

    def show(self):
        if hasattr(self.t_iter, 'show'):
            self.t_iter.show()

    def after_run(self, runner):
        self.t_iter.update(runner.max_iters)
        self.t_iter.on_iter_end()
        self.show()

    def log(self, runner):
        if not self.every_n_iters(runner, self.interval):
            return

        import torch
        if not self._title_inited:
            output_items_keys = list(runner.log_buffer.output.keys())
            items = ['iter'] + output_items_keys
            items += ['lr', 'max_mem']
            self.t_iter.names = output_items_keys
            self.t_iter.write(items, table=True)
            self._title_inited = True

        if torch.cuda.is_available():
            max_mem_mb = torch.cuda.max_memory_allocated() / 1024.0 / 1024.0
        else:
            max_mem_mb = None
        memory = "{:.0f}M".format(max_mem_mb) if max_mem_mb is not None else ""

        lr = '{:.6f}'.format(runner.current_lr()[0])

        iteration = runner.iter + 1
        values = [str(iteration)]
        values += ['{:.4f}'.format(v) for v in runner.log_buffer.output.values()]
        values += [lr, memory]
        self.t_iter.write(values, table=True)
        self.t_iter.update(iteration)
        self.show()
