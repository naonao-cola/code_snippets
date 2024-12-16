'''
python test_model_file.py -model_path ./models/65F/model.capp [-image_path=./images] [-sys_platform=lite/pro]
'''

import os
import argparse
# import unittest


def unpackage_capp(model_path):
    '''
    In:
        model_path: capp model path
    Out:
        model_dir: unpackage model path
    '''
    from zipfile import ZipFile, ZIP_DEFLATED
    if os.path.isfile(model_path):
        model_dir = model_path[:-5]
        os.makedirs(model_dir, exist_ok=True)
        zip_file = ZipFile(model_path, mode='r')
        zip_file.extractall(model_dir)
    else:
        model_dir = model_path
    return model_dir


class SysPlatform:
    lite = 'lite'
    pro = 'pro'


class CustomApp:
    def __init__(self, model_path):
        '''eg:
        '''
        import yaml
        self._model_path = model_path
        with open(os.path.join(model_path, 'model_info.yml'), 'rt') as fp:
            self._model_info = yaml.load(fp)
        import_cmd = self._model_info['import_inferece']
        print("import inferece cmd: ", import_cmd)
        # from xxx_algo import xxxInference
        exec(import_cmd)
        inf_cls = import_cmd.split()[-1]
        # xxxInference(model_path)
        self._app = eval(inf_cls)(model_path)
        assert self._app is not None, 'ERROR: init {inf_cls} failed'

    def get_class_list(self):
        return self._model_info['classes']

    def run(self, data_path, *args):
        '''eg:
        '''
        import glob
        image_list = glob.glob("%s/*.[jJpPbB][pPnNmM][gGpP]" % data_path)
        if len(image_list)==0:
            print('WARNING: len(image_list)==0. data_path:', data_path)
        if len(args) == 0:
            return self._app.run(image_list)
        else:
            return self._app.run(image_list, args[0])


def _check_result_pro(result):
    ''' check type, key, ...'''
    pass


def _check_result_lite(result):
    ''' check type, key, ...'''
    pass


def test_inference(model_path, data_path=None, platform=SysPlatform.pro):
    if model_path.split('.')[-1] == 'capp':
        print("unpackage_capp: ", model_path)
        model_dir = unpackage_capp(model_path)
    else:
        model_dir = model_path
    application = CustomApp(model_dir)
    ####
    res = None
    if data_path:
        res = application.run(data_path)
        print(res)
        if platform==SysPlatform.pro:
            _check_result_pro(res)
        elif platform==SysPlatform.lite:
            _check_result_lite(res)

    else:
        print("image_path is None, no exect application.run()")
    print('OK')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='run test model_file and xxxInrefence')
    parser.add_argument("-model_path", type=str, default="", help="model_path or model_file")
    parser.add_argument("-image_path", type=str, default=None, help="image path")
    parser.add_argument("-sys_platform", type=str, default='lite', help="TI sys platform, one of [lite, pro]")
    opt = parser.parse_args()
    test_inference(opt.model_path, opt.image_path)
