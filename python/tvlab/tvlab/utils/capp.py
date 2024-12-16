'''
Copyright (C) 2023 TuringVision

package/unpackage/load capp.
'''
import yaml
import time
import os
import os.path as osp


_MODEL_INFO_FILE = 'model_info.yml'

__all__ = ['package_capp', 'unpackage_capp', 'load_capp']


def package_capp(model_path, import_cmd, ext_info=None, pkg_files=None,
                 export_model_info=False):
    ''' package capp model
    model_path (str): capp model path

    import_cmd (str): import cmd for model inference (used by adc gpu server)

    ext_info (dict): extern info need to be saved in model_info
        eg: {'classes': ['A', 'B', ..],  'threshold': 0.85, ...}

    pkg_files (dict): Disk files that need to be packaged
        eg: {'model.pth': '/path/of/model.pth', 'cfg.yaml': '/path/of/cfg.yaml'}
    '''
    from uuid import getnode as get_mac
    from zipfile import ZipFile, ZIP_DEFLATED
    # 1. generate model_info.yml
    model_info = {
        'import_inferece': import_cmd,
        'date': time.asctime(),
        'description': 'MAC:' + str(get_mac()),
    }
    if ext_info is not None:
        model_info.update(ext_info)

    # 2. zip vision dir
    with ZipFile(model_path, 'w', ZIP_DEFLATED) as fp:
        fp.writestr(_MODEL_INFO_FILE, data=yaml.dump(model_info))
        if pkg_files:
            for k, v in pkg_files.items():
                fp.write(v, arcname=k)

    # 3. export model info to model_info_path if needed
    if export_model_info:
        model_info_path = osp.dirname(model_path)
        with open(osp.join(model_info_path, _MODEL_INFO_FILE), 'wt', encoding='utf-8') as f:
            yaml.dump(model_info, f)


def unpackage_capp(model_path):
    '''
    In:
        model_path: capp model path
    Out:
        model_dir: unpackage model path
    '''
    from zipfile import ZipFile, ZIP_DEFLATED
    if osp.isfile(model_path):
        model_dir = model_path[:-5]
        os.makedirs(model_dir, exist_ok=True)
        zip_file = ZipFile(model_path, mode='r')
        zip_file.extractall(model_dir)
    else:
        model_dir = model_path
    return model_dir


def load_capp(model_path, work_dir=None, cls=None):
    '''
    model_path: capp model path
    work_dir: work dir
    cls: Class for process this model
    '''
    model_dir = unpackage_capp(model_path)

    if cls is None:
        with open(osp.join(model_dir, _MODEL_INFO_FILE), 'rt', encoding='utf-8') as fp:
            model_info = yaml.load(fp, Loader=yaml.FullLoader)
            import_cmd = model_info['import_inferece']
            exec(import_cmd)
            cls = import_cmd.split()[-1]

    return eval(cls)(model_dir, work_dir=work_dir)
