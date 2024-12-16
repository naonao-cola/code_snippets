'''
Copyright (C) 2023 TuringVision

CameraServer
'''
from multiprocessing import Process, JoinableQueue, sharedctypes
from queue import Empty
from threading import Thread
import ctypes
import time
import logging
import numpy as np
from .gen_camera import GenCamera


__all__ = ['CameraServer']


logging.basicConfig(level = logging.INFO, format = '%(asctime)s[%(levelname)s]: %(message)s')
logger = logging.getLogger()

class CameraServer:
    '''
    Wrapper class for GenCamera, which run camera task in subprocess.
    Simple usage as below:
    1. Create CameraServer instance to initial devices.
        cam_info = {'MV-CE050-30GM(c4:2f:90:fa:d4:52)': {'Width':2592, 'Height':1944, 'Channel':1},...}
        cam = CameraServer(cam_info)
    2. List the available devices:
        device_list = cam.list_devices()
    3. Select one device to connect to.
        id_ = cam.connect_device(id_='MV-CE050-30GM(c4:2f:90:fa:d4:52)')
    4. Start preview, press [close] button to exit:
        cam.start_preview([id_])
    5. Get the available properties of the connected device.
        cam.get_all_properties(id_)
    6. Set any property:
        cam.set_property(id_, 'Height', 1024)
        cam.set_property(id_, 'Width', 768)
    7. Register image callback.
        cam.register_img_cb(id_, cb)
    8. Start grabbing the stream.
        cam.start_grabbing(id_)
    9. Get one frame image:
        img = cam.get_one_frame(id_)
    10. Stop grabbing.
        cam.stop_grabbing(id_)
    11. Disconnect device:
        cam.disconnect_device(id_)
    12. Dump device properties to json file.
        cam.dump_properties(filepath, id_)
    13. Load device properties from json file.
        cam.load_properties(filepath, id_)
    '''
    def __init__(self, dev_infos):
        '''
        dev_infos: the image size of the device. used to create sharememory.
            e.g.
            cam_info = {'MV-CE050-30GM(c4:2f:90:fa:d4:52)': {'Width':2592, 'Height':1944, 'Channel':1},...}
        '''
        self.req_queue = JoinableQueue()
        self.res_queue = JoinableQueue()
        self.devices = {}
        for id_, info in dev_infos.items():
            imgsize = info['Width'] * info['Height'] * info['Channel']
            self.devices[id_] = {}
            self.devices[id_]['w'] = info['Width']
            self.devices[id_]['h'] = info['Height']
            self.devices[id_]['c'] = info['Channel']
            self.devices[id_]['shm'] = sharedctypes.RawArray(ctypes.c_ubyte, imgsize)
            self.devices[id_]['img_cbs'] = []
            self.devices[id_]['cb_thread'] = None
            self.devices[id_]['grabing'] = False
            self.devices[id_]['img_queue'] = JoinableQueue()

        self._start_server()
        logger.info('CameraServer init')

    def _start_server(self):
        self.process = Process(target=self._cam_process_func, daemon=True, args=(self.req_queue, self.res_queue))
        self.process.start()
        logger.info('camera server started!')

    def _cb_forward_thread_func(self, id_):
        while self.devices[id_]['grabing']:
            try:
                self.devices[id_]['img_queue'].get(True, 0.1)
            except Empty:
                continue
            img_cbs = self.devices[id_]['img_cbs']
            if len(img_cbs) == 0:
                continue
            w = self.devices[id_]['w']
            h = self.devices[id_]['h']
            c = self.devices[id_]['c']
            img = np.frombuffer(self.devices[id_]['shm'], dtype=np.uint8, count=w*h*c).reshape(h,w,c)
            img = img.copy()
            for cb in img_cbs:
                cb(id_, img)
            # logger.debug(f'Callback done: cost:{time.time() - data[0]}  shape:{img.shape}')
            self.devices[id_]['img_queue'].task_done()

    def _internal_image_cb(self, id_, img):
        t1 = time.time()
        np_arr = np.frombuffer(self.devices[id_]['shm'], dtype=np.uint8, count=np.prod(img.shape))
        np_arr.shape = img.shape
        np_arr[...] = img
        self.devices[id_]['img_queue'].put([t1])
        self.devices[id_]['img_queue'].join()

    def _cam_process_func(self, req_queue, res_queue):
        self.cam = GenCamera()
        while True:
            try:
                req = req_queue.get(True)
                methon = f'self.cam.{req["method"]}'
                args = req['args']
                kws = req['kws']

                if req["method"] in ['register_image_cb', 'unregister_image_cb']:
                    args = list(args)
                    args.append(self._internal_image_cb)

                if bool(kws):
                    ret = eval(methon)(**kws)
                elif bool(args):
                    ret = eval(methon)(*args)
                else:
                    ret = eval(methon)()
                req_queue.task_done()

                res_queue.put(ret)
                res_queue.join()
            except Exception as e:
                logger.exception(e)

    def _make_method_req(self, method, args=None, kws=None):
        return {'method':method, 'args':args, 'kws':kws}

    def _run_method_in_process(self, method_req):
        # print('[Client] method_req:',method_req)
        self.req_queue.put(method_req)
        self.req_queue.join()
        ret = self.res_queue.get(True)
        self.res_queue.task_done()
        # print(f'[Client] method_req:{method_req}  RET:{ret}')
        return ret

    def list_devices(self):
        '''List the infomation of available devices.
        '''
        method_req = self._make_method_req('list_devices')
        return self._run_method_in_process(method_req)

    def connect_device(self, list_index=0, **kws):
        '''
        list_index: The index of device_info_list. Note: the first one in the device list is connected by default.
        kws: Information used to match the unique device.
        e.g. connect_device(vendor='Daheng Imaging', serial_number='QY0180030001')
        '''
        method_req = self._make_method_req('connect_device', (list_index,), kws)
        return self._run_method_in_process(method_req)

    def get_property(self, id_, name):
        '''name: The name for get property. call get_all_properties() to get the available properties.
        '''
        method_req = self._make_method_req('get_property', (id_, name))
        return self._run_method_in_process(method_req)

    def set_property(self, id_, name, value):
        '''name: The name for set property. call get_all_properties() to get the available properties.
        '''
        method_req = self._make_method_req('set_property', (id_, name, value))
        self._run_method_in_process(method_req)

    def get_all_properties(self, id_):
        '''Return all the available properies of current connected device.
        '''
        method_req = self._make_method_req('get_all_properties', (id_,))
        return self._run_method_in_process(method_req)

    def register_image_cb(self, id_, img_cb):
        '''Register a img_cb to receive the img in numpy array format.
            id_: device id, returned by connect_device()
            img_cb: will be called when an image is ready.
            e.g. def img_cb(id_, img):
                    print(id_, img.shape)
        '''
        img_cbs = self.devices[id_]['img_cbs']
        img_cbs.append(img_cb)

        if len(img_cbs) == 1:
            method_req = self._make_method_req('register_image_cb', (id_,))
            self._run_method_in_process(method_req)

    def unregister_image_cb(self, id_, img_cb=None):
        '''Register a img_cb to receive the img in numpy array format.
        ia: image acquisition object, returned by connect_device()
        img_cb: callback to unregister
        '''
        if img_cb is None:
            self.devices[id_]['img_cbs'] = []
        elif img_cb in self.devices[id_]['img_cbs']:
            self.devices[id_]['img_cbs'].remove(img_cb)

        if len(self.devices[id_]['img_cbs']) == 0:
            method_req = self._make_method_req('unregister_image_cb', (id_,))
            self._run_method_in_process(method_req)

    def start_grabbing(self, id_):
        '''Start grabbing image stream, after that,
           you can get images via regisgered img_cb or call cam.get_one_frame()
        '''
        method_req = self._make_method_req('start_grabbing', (id_,))
        self._run_method_in_process(method_req)
        self.devices[id_]['grabing'] = True
        thread = Thread(target=self._cb_forward_thread_func, daemon=True, args=(id_,))
        thread.start()
        self.devices[id_]['cb_thread'] = thread

    def stop_grabbing(self, id_, sync=True):
        '''Stop grabbing image stream.
            ia: image acquisition object, returned by connect_device()
            sync: wait until the grabbing thread exited
        '''
        method_req = self._make_method_req('stop_grabbing', (id_, sync))
        self._run_method_in_process(method_req)

        self.devices[id_]['grabing'] = False
        if sync and self.devices[id_]['cb_thread']:
            self.devices[id_]['cb_thread'].join()
            self.devices[id_]['cb_thread'] = None

    def get_one_frame(self, id_, timeout=100):
        ''' get one frame image from grabbing stream.
        In:
            ia: image acquisition object, returned by connect_device()
            timeout: Less or equal 0 for infinite wait.
        Out:
            numpy image
        '''
        method_req = self._make_method_req('get_one_frame', (id_, timeout))
        img = self._run_method_in_process(method_req)
        return img

    def software_trigger(self, id_):
        '''
        software trigger device to capture one image.
        refer below to let device work in software trigger mode:
            cam.set_property(id_, "TriggerMode", "On")
            cam.get_property(id_, "TriggerSource")
        '''
        method_req = self._make_method_req('software_trigger', (id_,))
        self._run_method_in_process(method_req)

    def dump_properties(self, filepath, id_=None):
        '''
        Dump device properties to a json file.
        filepath: json file path
        id_: device id, if not set. will dump properties for all the connected devices.
        '''
        method_req = self._make_method_req('dump_properties', (filepath, id_,))
        self._run_method_in_process(method_req)

    def load_properties(self, filepath, id_=None):
        '''
        Load device properties from a json file.
        filepath: json file path
        id_: device id, if not set. will load properties for
             all the connected devices from the json file if the id_ is match.
        '''
        method_req = self._make_method_req('load_properties', (filepath, id_,))
        self._run_method_in_process(method_req)

    def start_preview(self, id_list, win_width=700, modify_func=None):
        ''' start camera preview.
        In:
            ia_list: image acquisition object list, can preview multi-cameras in same time.
            win_width: preview window width.
            modify_func: the function used to modify preview frame image.
                def modify_func(img):
                    #do some modify for img
                    return img
        '''
        method_req = self._make_method_req('start_preview', (id_list, win_width, modify_func))
        self._run_method_in_process(method_req)


if __name__ == '__main__':
    cam_info = {
        'MV-CE050-30GM(c4:2f:90:fa:d4:52)': {'Width':2592, 'Height':1944, 'Channel':1},
        'LBAS-GE50-23C(c4:2f:90:fe:b7:74)': {'Width':2448, 'Height':2048, 'Channel':1}
    }
    cam = CameraServer(cam_info)
    # cam = GenCamera()
    devinfo_list = cam.list_devices()
    id1 = cam.connect_device(id_='MV-CE050-30GM(c4:2f:90:fa:d4:52)')
    id2 = cam.connect_device(id_='LBAS-GE50-23C(c4:2f:90:fe:b7:74)')

    cam.set_property(id1, "TriggerMode", "Off")
    cam.set_property(id2, "TriggerMode", "Off")
    cam.get_property(id1, "TriggerSource")
    cam.get_property(id2, "TriggerSource")

    cam.set_property(id1, 'Height', 1944)
    cam.set_property(id1, 'Width', 2592)
    cam.set_property(id2, 'Height', 2048)
    cam.set_property(id2, 'Width', 2448)

    #启动preview
    # cam.start_preview([id1])

    #注册回调
    def image_callback(id_, img):
        print(f'image_callback, id_={id_}  shape={img.shape}')
    cam.register_image_cb(id1, image_callback)
    cam.register_image_cb(id2, image_callback)

    # #开始取图
    cam.start_grabbing(id1)
    cam.start_grabbing(id2)

    time.sleep(3)
    cam.stop_grabbing(id1)
    cam.stop_grabbing(id2)
