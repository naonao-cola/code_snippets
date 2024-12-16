'''
Copyright (C) 2023 TuringVision
'''
import os
import time
import yaml
from copy import deepcopy
import numpy as np
import paddleocr
import sys


class AttrDict(dict):
    """Single level attribute dict, NOT recursive"""

    def __init__(self, **kwargs):
        super(AttrDict, self).__init__()
        super(AttrDict, self).update(kwargs)

    def __getattr__(self, key):
        if key in self:
            return self[key]
        raise AttributeError("object has no attribute '{}'".format(key))


PRE_URLS = {
    "MobileNetV3_large_x0_5_pretrained": {
        "url": "https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_large_x0_5_pretrained.tar",
        "isDir": True,
    },
    "ResNet50_vd_ssld_pretrained": {
        "url": "https://paddle-imagenet-models-name.bj.bcebos.com/ResNet50_vd_ssld_pretrained.tar",
        "isDir": True,
    },
    "rec_mv3_none_bilstm_ctc": {
        "url": "https://paddleocr.bj.bcebos.com/rec_mv3_none_bilstm_ctc.tar",
        "isDir": True,
    },
    "ch_ppocr_mobile_v1.1_rec_pre": {
        "url": "https://paddleocr.bj.bcebos.com/20-09-22/mobile/rec/ch_ppocr_mobile_v1.1_rec_pre.tar",
        "isDir": False
    }
}

global_config = AttrDict()

default_config = {'Global': {'debug': False, }}


def load_config(file_path):
    """
    Load config from yml/yaml file.
    Args:
        file_path (str): Path of the config file to be loaded.
    Returns: global config
    """
    merge_config(default_config)
    _, ext = os.path.splitext(file_path)
    assert ext in ['.yml', '.yaml'], "only support yaml files for now"
    merge_config(yaml.load(open(file_path, 'rb'), Loader=yaml.Loader))
    if "reader_yml" in global_config['Global'] and global_config['Global']["reader_yml"] is not None and os.path.exists(global_config['Global']['reader_yml']):
        assert "reader_yml" in global_config['Global'],\
            "absence reader_yml in global"
        reader_file_path = global_config['Global']['reader_yml']
        _, ext = os.path.splitext(reader_file_path)
        assert ext in ['.yml', '.yaml'], "only support yaml files for reader"
        merge_config(yaml.load(open(reader_file_path, 'rb'), Loader=yaml.Loader))
    return global_config


def merge_config(config):
    """
    Merge config into global config.
    Args:
        config (dict): Config to be merged.
    Returns: global config
    """
    for key, value in config.items():
        if "." not in key:
            if isinstance(value, dict) and key in global_config:
                global_config[key].update(value)
            else:
                global_config[key] = value
        else:
            sub_keys = key.split('.')
            assert (
                sub_keys[0] in global_config
            ), "the sub_keys can only be one of global_config: {}, but get: {}, please check your running command".format(
                global_config.keys(), sub_keys[0])
            cur = global_config[sub_keys[0]]
            for idx, sub_key in enumerate(sub_keys[1:]):
                assert (
                    sub_key in cur
                ), "key {} not in sub_keys: {}, please check your running command.".format(
                    sub_key, cur)
                if idx == len(sub_keys) - 2:
                    cur[sub_key] = value
                else:
                    cur = cur[sub_key]


def check_gpu(use_gpu):
    """
    Log error and exit when set use_gpu=true in paddlepaddle
    cpu version.
    """
    from paddle import fluid
    err = "Config use_gpu cannot be set as true while you are " \
          "using paddlepaddle cpu version ! \nPlease try: \n" \
          "\t1. Install paddlepaddle-gpu to run model on GPU \n" \
          "\t2. Set use_gpu as false in config file to run " \
          "model on CPU"

    try:
        if use_gpu and not fluid.is_compiled_with_cuda():
            print(err)
            sys.exit(1)
    except Exception as e:
        pass


def build(config, main_prog, startup_prog, mode):
    """
    Build a program using a model and an optimizer
        1. create a dataloader
        2. create a model
        3. create fetches
        4. create an optimizer
    Args:
        config(dict): config
        main_prog(): main program
        startup_prog(): startup program
        mode(str): train or valid
    Returns:
        dataloader(): a bridge between the model and the data
        fetch_name_list(dict): dict of model outputs(included loss and measures)
        fetch_varname_list(list): list of outputs' varname
        opt_loss_name(str): name of loss
    """
    from paddle import fluid
    from ppocr.utils.utility import create_module

    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            func_infor = config['Architecture']['function']
            model = create_module(func_infor)(params=config)
            dataloader, outputs = model(mode=mode)
            fetch_name_list = list(outputs.keys())
            fetch_varname_list = [outputs[v].name for v in fetch_name_list]
            opt_loss_name = None
            model_average = None
            img_loss_name = None
            word_loss_name = None
            if mode == "train":
                opt_loss = outputs['total_loss']
                # srn loss
                #img_loss = outputs['img_loss']
                #word_loss = outputs['word_loss']
                #img_loss_name = img_loss.name
                #word_loss_name = word_loss.name
                opt_params = config['Optimizer']
                optimizer = create_module(opt_params['function'])(opt_params)
                optimizer.minimize(opt_loss)
                opt_loss_name = opt_loss.name
                global_lr = optimizer._global_learning_rate()
                fetch_name_list.insert(0, "lr")
                fetch_varname_list.insert(0, global_lr.name)
                if "loss_type" in config["Global"]:
                    if config['Global']["loss_type"] == 'srn':
                        model_average = fluid.optimizer.ModelAverage(
                            config['Global']['average_window'],
                            min_average_window=config['Global'][
                                'min_average_window'],
                            max_average_window=config['Global'][
                                'max_average_window'])

    return (dataloader, fetch_name_list, fetch_varname_list, opt_loss_name,
            model_average)


def build_export(config, main_prog, startup_prog):
    """
    Build input and output for exporting a checkpoints model to an inference model
    Args:
        config(dict): config
        main_prog: main program
        startup_prog: startup program
    Returns:
        feeded_var_names(list[str]): var names of input for exported inference model
        target_vars(list[Variable]): output vars for exported inference model
        fetches_var_name: dict of checkpoints model outputs(included loss and measures)
    """
    from paddle import fluid
    from ppocr.utils.utility import create_module

    with fluid.program_guard(main_prog, startup_prog):
        with fluid.unique_name.guard():
            func_infor = config['Architecture']['function']
            model = create_module(func_infor)(params=config)
            algorithm = config['Global']['algorithm']
            if algorithm == "SRN":
                image, others, outputs = model(mode='export')
            else:
                image, outputs = model(mode='export')
            fetches_var_name = sorted([name for name in outputs.keys()])
            fetches_var = [outputs[name] for name in fetches_var_name]
    if algorithm == "SRN":
        others_var_names = sorted([name for name in others.keys()])
        feeded_var_names = [image.name] + others_var_names
    else:
        feeded_var_names = [image.name]

    target_vars = fetches_var
    return feeded_var_names, target_vars, fetches_var_name


def create_multi_devices_program(program, loss_var_name, for_quant=False):
    from paddle import fluid
    build_strategy = fluid.BuildStrategy()
    build_strategy.memory_optimize = False
    build_strategy.enable_inplace = True
    if for_quant:
        build_strategy.fuse_all_reduce_ops = False
    else:
        program = fluid.CompiledProgram(program)
    exec_strategy = fluid.ExecutionStrategy()
    exec_strategy.num_iteration_per_drop_scope = 1
    compile_program = program.with_data_parallel(
        loss_name=loss_var_name,
        build_strategy=build_strategy,
        exec_strategy=exec_strategy)
    return compile_program


def cal_det_res(exe, config, eval_info_dict):
    from ppocr.utils.utility import create_module
    from fastprogress.fastprogress import progress_bar
    global_params = config['Global']
    postprocess_params = deepcopy(config["PostProcess"])
    postprocess_params.update(global_params)
    postprocess = create_module(postprocess_params['function']) \
        (params=postprocess_params)
    if config['Global']['algorithm'] == 'DB':
        from .db_process import DBPostProcessTS
        postprocess = DBPostProcessTS(params=postprocess_params)

    results = []
    gt_list = []
    tackling_num = 0
    for data in progress_bar(eval_info_dict['reader']()):
        img_num = len(data)
        tackling_num = tackling_num + img_num
        img_list = []
        ratio_list = []
        for ino in range(img_num):
            img_list.append(data[ino][0])
            ratio_list.append(data[ino][1])
            gt_list.append(data[ino][2])
        try:
            img_list = np.concatenate(img_list, axis=0)
        except:
            err = "concatenate error usually caused by different input image shapes in evaluation or testing.\n \
            Please set \"test_batch_size_per_card\" in main yml as 1\n \
            or add \"test_image_shape: [h, w]\" in reader yml for EvalReader."

            raise Exception(err)
        outs = exe.run(eval_info_dict['program'], \
                        feed={'image': img_list}, \
                        fetch_list=eval_info_dict['fetch_varname_list'])
        outs_dict = {}
        for tno in range(len(outs)):
            fetch_name = eval_info_dict['fetch_name_list'][tno]
            fetch_value = np.array(outs[tno])
            outs_dict[fetch_name] = fetch_value
        dt_boxes_list, dt_scores_list = postprocess(outs_dict, ratio_list)
        for ino in range(img_num):
            dt_boxes = dt_boxes_list[ino]
            dt_scores = dt_scores_list[ino]
            item = {'labels': [], 'polygons': [],}
            for idx, box in enumerate(dt_boxes):
                item["labels"].append("")
                tpp = np.array(box).reshape((-1, )).tolist()
                tpp.append(dt_scores[idx])
                item["polygons"].append(tpp)
            results.append(item)

    return results, gt_list

def format_data(gts):
    yture = []
    for item in gts:
        one_img = {"labels":[], "polygons":[]}
        for word in item:
            one_img["labels"].append("object")
            one_img["polygons"].append(word["points"])
        yture.append(one_img)
    return yture

def eval_det_run(exe, config, eval_info_dict, mode):
    from ..segmentation.eval_segmentation import EvalSegmentation
    from ..segmentation.polygon_label import PolygonLabelList
    ypred, ytrue = cal_det_res(exe, config, eval_info_dict)
    ytrue = format_data(ytrue)
    evals = EvalSegmentation(ypred, ytrue, polygons_only=True)
    tmp = evals.get_result()['object']

    metrics = {
        "hmean": tmp["f1"],
        "precision": tmp["precision"],
        "recall": tmp["recall"],
    }
    return evals, metrics


def train_eval_det_run(config,
                       exe,
                       train_info_dict,
                       eval_info_dict,
                       callback=None,
                       is_slim=None):
    """
    Feed data to the model and fetch the measures and loss for detection
    Args:
        config: config
        exe:
        train_info_dict: information dict for training
        eval_info_dict: information dict for evaluation
    """
    from paddle import fluid
    from ppocr.utils.stats import TrainingStats
    from ppocr.utils.save_load import save_model

    train_batch_id = 0
    log_smooth_window = config['Global']['log_smooth_window']
    epoch_num = config['Global']['epoch_num']
    print_batch_step = config['Global']['print_batch_step']
    eval_batch_step = config['Global']['eval_batch_step']
    start_eval_step = 0
    if type(eval_batch_step) == list and len(eval_batch_step) >= 2:
        start_eval_step = eval_batch_step[0]
        eval_batch_step = eval_batch_step[1]
        print(
            "During the training process, after the {}th iteration, an evaluation is run every {} iterations".
            format(start_eval_step, eval_batch_step))
    save_epoch_step = config['Global']['save_epoch_step']
    save_model_dir = config['Global']['save_model_dir']
    if not os.path.exists(save_model_dir):
        os.makedirs(save_model_dir)
    train_stats = TrainingStats(log_smooth_window,
                                train_info_dict['fetch_name_list'])
    best_eval_hmean = -1
    best_eval_all = None
    best_batch_id = 0
    best_epoch = 0
    train_loader = train_info_dict['reader']
    for epoch in range(epoch_num):
        train_loader.start()
        try:
            while True:
                flag = False
                metrics = None
                t1 = time.time()
                train_outs = exe.run(
                    program=train_info_dict['compile_program'],
                    fetch_list=train_info_dict['fetch_varname_list'],
                    return_numpy=False)
                stats = {}
                for tno in range(len(train_outs)):
                    fetch_name = train_info_dict['fetch_name_list'][tno]
                    fetch_value = np.mean(np.array(train_outs[tno]))
                    stats[fetch_name] = fetch_value
                t2 = time.time()
                train_batch_elapse = t2 - t1
                train_stats.update(stats)
                if train_batch_id > 0 and train_batch_id  \
                    % print_batch_step == 0:
                    flag = True
                    logs = train_stats.log()
                    strs = 'epoch: {}, iter: {}, {}, time: {:.3f}'.format(
                        epoch, train_batch_id, logs, train_batch_elapse)
                    print(strs)


                if train_batch_id > start_eval_step and\
                    (train_batch_id - start_eval_step) % eval_batch_step == 0  or train_batch_id == 1:
                    flag = True
                    _, metrics = eval_det_run(exe, config, eval_info_dict, "eval")
                    hmean = metrics['hmean']
                    if hmean >= best_eval_hmean:
                        best_eval_all = metrics
                        best_eval_hmean = hmean
                        best_batch_id = train_batch_id
                        best_epoch = epoch
                        save_path = save_model_dir + "/best_accuracy"
                        if is_slim is None:
                            save_model(train_info_dict['train_program'],
                                       save_path)
                        else:
                            import paddleslim as slim
                            if is_slim == "prune":
                                slim.prune.save_model(
                                    exe, train_info_dict['train_program'],
                                    save_path)
                            elif is_slim == "quant":
                                save_model(eval_info_dict['program'], save_path)
                            else:
                                raise ValueError(
                                    "Only quant and prune are supported currently. But received {}".
                                    format(is_slim))
                    strs = 'Test iter: {}, metrics:{}, best_hmean:{:.6f}, best_epoch:{}, best_batch_id:{}'.format(
                        train_batch_id, metrics, best_eval_hmean, best_epoch,
                        best_batch_id)
                    print(strs)

                if flag and callback is not None:
                    temp_stats = train_stats.get()
                    status = {
                        'desc': 'training_epoch{}_iter{}'.format(epoch, train_batch_id),
                        'percent': int(100 * (epoch + 0.00001) / (epoch_num + 0.00001)),
                        'epoch': epoch,
                        'epochs': epoch_num,
                        'pass_desc': 'epoch_{}'.format(epoch_num),
                        'loss': temp_stats["total_loss"],
                        'addition': {
                            'iter': train_batch_id,
                            'lr': temp_stats["lr"],
                            'loss_shrink_maps': temp_stats["loss_shrink_maps"],
                            'loss_threshold_maps': temp_stats["loss_threshold_maps"],
                            'loss_binary_maps': temp_stats["loss_binary_maps"],
                        }
                    }
                    if metrics is not None:
                        status["addition"]["precision"] = metrics["precision"]
                        status["addition"]["recall"] = metrics["recall"]
                        status["addition"]["hmean"] = metrics["hmean"]

                    callback(status)

                if callback is not None and callback({'desc': 'training_step', 'iter':train_batch_id, 'max_iter':None}):
                    raise KeyboardInterrupt("Stop train successful!")

                train_batch_id += 1

        except fluid.core.EOFException:
            train_loader.reset()
        if epoch == 0 and save_epoch_step == 1:
            save_path = save_model_dir + "/iter_epoch_0"
            if is_slim is None:
                save_model(train_info_dict['train_program'], save_path)
            else:
                import paddleslim as slim
                if is_slim == "prune":
                    slim.prune.save_model(exe, train_info_dict['train_program'],
                                          save_path)
                elif is_slim == "quant":
                    save_model(eval_info_dict['program'], save_path)
                else:
                    raise ValueError(
                        "Only quant and prune are supported currently. But received {}".
                        format(is_slim))
        if epoch > 0 and epoch % save_epoch_step == 0:
            save_path = save_model_dir + "/iter_epoch_%d" % (epoch)
            if is_slim is None:
                save_model(train_info_dict['train_program'], save_path)
            else:
                import paddleslim as slim
                if is_slim == "prune":
                    slim.prune.save_model(exe, train_info_dict['train_program'],
                                          save_path)
                elif is_slim == "quant":
                    save_model(eval_info_dict['program'], save_path)
                else:
                    raise ValueError(
                        "Only quant and prune are supported currently. But received {}".
                        format(is_slim))
    return best_eval_all


def load_model_from_url(config):
    import subprocess
    pretrain_weights = config["Global"]["pretrain_weights"]
    if pretrain_weights and not os.path.exists(pretrain_weights) and not os.path.exists(pretrain_weights+".pdparams"):
        print('Need load pretrain_weights......')
        splited_path = pretrain_weights.split("/")
        if splited_path[-1] == "" or "best_accuracy" in splited_path[-1]:
            base_name = splited_path[-2]
            pretrain_dir = "/".join(splited_path[:-2])
        else:
            base_name = splited_path[-1]
            pretrain_dir = "/".join(splited_path[:-1])

        cfg_msg = PRE_URLS[base_name]
        url = cfg_msg["url"]
        os.makedirs(pretrain_dir, exist_ok=True)
        cmd1 = "wget -P {} {}".format(pretrain_dir, url)
        if cfg_msg["isDir"]:
            cmd2 = "cd {} && tar xf {}".format(pretrain_dir, "{}.tar".format(base_name))
        else:
            cmd2 = "cd {} && tar xf {} && mv {} {}.pdparams".format(pretrain_dir, "{}.tar".format(base_name), base_name, base_name)
        try:
            print("Start loading {}".format(url))
            subprocess.check_output(cmd1, shell=True)
            subprocess.check_output(cmd2, shell=True)
            print("loading success ......")
        except:
            print("Maybe url error...")
            print("exe commad fail: \n {}\n {} \n".format(cmd1, cmd2))

def preprocess(work_dir, cfg_file):
    from paddle import fluid
    # load config from yml file
    config = load_config(cfg_file)
    config["Global"]["save_model_dir"] = work_dir

    # load pretrained model
    load_model_from_url(config)

    # check if set use_gpu=True in paddlepaddle cpu version
    use_gpu = config['Global']['use_gpu']
    check_gpu(use_gpu)

    # check whether the set algorithm belongs to the supported algorithm list
    alg = config['Global']['algorithm']
    assert alg in [
        'EAST', 'DB', 'SAST', 'Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN', 'CLS'
    ]
    if alg in ['Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN']:
        from ppocr.utils.character import CharacterOps
        config['Global']['char_ops'] = CharacterOps(config['Global'])

    place = fluid.CUDAPlace(0) if use_gpu else fluid.CPUPlace()
    startup_program = fluid.Program()
    train_program = fluid.Program()

    if alg in ['EAST', 'DB', 'SAST']:
        train_alg_type = 'det'
    elif alg in ['Rosetta', 'CRNN', 'STARNet', 'RARE', 'SRN']:
        train_alg_type = 'rec'
    else:
        train_alg_type = 'cls'

    return startup_program, train_program, place, config, train_alg_type
