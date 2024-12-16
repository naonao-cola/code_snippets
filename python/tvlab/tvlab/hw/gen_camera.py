'''
Copyright (C) 2023 TuringVision

GenCamera
'''

import threading
import os
import sys
import copy
import logging

__all__ = ['GenCamera']

logging.basicConfig(level = logging.INFO, format = '%(asctime)s[%(levelname)s]: %(message)s')
logger = logging.getLogger()

class GenCamera:
    '''
    Generic Camera module for aquiring images.
    Simple usage as below:
    1. Create GenCamera instance to initial devices.
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
    def __init__(self):
        pass

    def __del__(self):
        if hasattr(self, 'harvester'):
            self.harvester.reset()

    def __new__(cls, *args, **kw):
        '''
        Use single instance to avoid confusion of device status
        '''
        if not hasattr(cls, '_instance'):
            orig = super(GenCamera, cls)
            cls._instance = orig.__new__(cls, *args, **kw)
        else:
            cls._instance.harvester.reset()

        cls._instance._check_env()
        cls._instance._init_devices()
        return cls._instance

    def _check_env(self):
        try:
            from harvesters.core import Harvester
            from harvesters.util.pfnc import mono_location_formats
        except Exception:
            print("Error: [harvesters] package not found.")
            print("You can install it by:")
            print("    1. pip install harvesters")
            print("    2. download and build it from https://github.com/genicam/harvesters")
            sys.exit()

        mv_impact_home = os.getenv('MV_IMPACT_HOME')
        if mv_impact_home == None:
            print('Error: MV_IMPACT_HOME not found！')
            print('GenericCamera need install a GenTL Producer to support image acquisition.')
            print('The recomended GenTL Producer is mvIMPACT_Acquire from MATRIX VISION,')
            print('Please download and install it from: http://static.matrix-vision.com/mvIMPACT_Acquire/2.29.0/')
            print('Or get it from SVN: ../04-Industry_Algorithm/02-内部资料/02-相机驱动/通用相机驱动/mvIMPACT_Acquire')
            print('Linux:')
            print('    1. Download install_mvGenTL_Acquire.sh and mvGenTL_Acquire-x86_64_ABI2-2.29.0.tgz')
            print('    2. ./install_mvGenTL_Acquire.sh')
            print('Windows:')
            print('    1. Download and install mvGenTL_Acquire-x86_64-2.29.0.msi\n')
            print('And then set MV_IMPACT_HOME=(PATH_TO_mvIMPACT)/')
            print('    e.g.export MV_IMPACT_HOME=/opt/mvIMPACT_Acquire')
            sys.exit()

    def _init_devices(self):
        from harvesters.core import Harvester
        self.harvester = Harvester()
        self.connect_ias = {}
        self.img_cbs = {} #key:ia.device.id_, val: cb list

        mv_impact_home = os.getenv('MV_IMPACT_HOME')
        subdir = 'bin/x64/' if os.name == 'nt' else 'lib/x86_64/'
        cti_file = os.path.join(mv_impact_home, subdir+'mvGenTLProducer.cti')
        self.harvester.add_file(cti_file)
        self.harvester.update()

    def list_devices(self):
        '''List the infomation of available devices.
        '''
        devinfo_list = []
        for dev in self.harvester.device_info_list:
            dev_info = {}
            dev = dev.__repr__()[1:-1].replace('\'', '')
            attrs = dev.split(', ')
            for attr in attrs:
                kv = attr.split('=')
                dev_info[kv[0]]=kv[1]
            devinfo_list.append(dev_info)
        return devinfo_list

    def connect_device(self, list_index=0, **kws):
        '''
        list_index: The index of device_info_list. Note: the first one in the device list is connected by default.
        kws: Information used to match the unique device.
        e.g. connect_device(vendor='Daheng Imaging', serial_number='QY0180030001')
        '''
        if len(self.harvester.device_info_list) == 0:
            raise Exception('Can not connect to device. No device available.')

        from genicam.gentl import AccessDeniedException, IoException
        try:
            ia = self.harvester.create_image_acquirer(**kws) if bool(kws) else self.harvester.create_image_acquirer(list_index)
        except AccessDeniedException:
            logger.error('AccessDeniedException occured, Device maybe in use!')
            sys.exit()
        except IoException:
            logger.error('IoException occured, Please check the IP settings, Camera and PC should be on the same network segment!')
            sys.exit()

        ia.ia_thread = None
        self.connect_ias[ia.device.id_] = ia

        logger.info(f'Connect to Device: {ia.device.id_}')
        return ia.device.id_

    def disconnect_device(self, id_=None):
        ias_to_disconnect = [self.connect_ias[id_]] if id_ != None else self.connect_ias.values()
        for ia in ias_to_disconnect:
            if ia.is_acquiring():
                ia.stop_acquisition()
            ia.destroy()
            self.connect_ias.pop(id_)
            logger.info(f'Disconnect_device: {ia.device.id_}')

    def get_property(self, id_, name):
        '''name: The name for get property. call get_all_properties() to get the available properties.
        '''
        try:
            return getattr(self.connect_ias[id_].remote_device.node_map, name).value
        except:
            logger.warning('Property [{}] does not appear to exist'.format(name))

    def set_property(self, id_, name, value):
        '''name: The name for set property. call get_all_properties() to get the available properties.
        '''
        try:
            getattr(self.connect_ias[id_].remote_device.node_map, name).value = value
        except Exception:
            logger.warning('Property [{}] does not appear to exist or is not writable.'.format(name))

    def get_all_properties(self, id_):
        '''Return all the available properies of current connected device.
        '''
        ia = self.connect_ias[id_]
        attr_list = dir(ia.remote_device.node_map)
        prop_dict = {}
        for attr in attr_list:
            try:
                if attr == 'TLParamsLocked':
                    continue
                prop_dict[attr] = getattr(ia.remote_device.node_map, attr).value
            except:
                pass
        return prop_dict

    def register_image_cb(self, id_, img_cb):
        '''Register a img_cb to receive the img in numpy array format.
            id_: device id, returned by connect_device()
            img_cb: will be called when an image is ready.
            e.g. def img_cb(id_, img):
                    print(id_, img.shape)
        '''
        ia = self.connect_ias[id_]
        if ia.device.id_ in self.img_cbs.keys():
            self.img_cbs[ia.device.id_].append(img_cb)
        else:
            self.img_cbs[ia.device.id_] = [img_cb]

    def unregister_image_cb(self, id_, img_cb=None):
        '''Register a img_cb to receive the img in numpy array format.
        ia: image acquisition object, returned by connect_device()
        img_cb: callback to unregister
        '''
        ia = self.connect_ias[id_]
        assert ia.device.id_ in self.img_cbs.keys()
        if img_cb is None:
            self.img_cbs.pop(ia.device.id_)
        else:
            self.img_cbs[ia.device.id_].remove(img_cb)

    def start_grabbing(self, id_):
        '''Start grabbing image stream, after that, you can get images via regisgered img_cb or call cam.get_one_frame()
        '''
        ia = self.connect_ias[id_]
        if ia.is_acquiring():
            logger.error(f'[{ia.device.id_}] Another grabbing task is running!')
        else:
            ia.start_acquisition(run_in_background=False)
            if len(self.img_cbs.keys()) > 0:
                ia.ia_thread = threading.Thread(target=self._grabbing_thread_func, args=(ia, self.img_cbs,))
                ia.lock = threading.Lock()
                ia.ia_thread.start()

        logger.info(f'[{ia.device.id_}] Start grabbing.')


    def stop_grabbing(self, id_, sync=True):
        '''Stop grabbing image stream.
            ia: image acquisition object, returned by connect_device()
            sync: wait until the grabbing thread exited
        '''
        ia = self.connect_ias[id_]
        if ia.is_acquiring():
            if ia.ia_thread:
                with ia.lock:
                    ia.stop_acquisition()
            else:
                ia.stop_acquisition()
        if sync and ia.ia_thread:
            ia.ia_thread.join()
        logger.info(f'[{ia.device.id_}] Stop grabbing.')

    def get_one_frame(self, id_, timeout=100):
        ''' get one frame image from grabbing stream.
        In:
            ia: image acquisition object, returned by connect_device()
            timeout: Less or equal 0 for infinite wait.
        Out:
            numpy image
        '''
        ia = self.connect_ias[id_]
        if not ia.is_acquiring():
            logger.error('[{ia.device.id_}] Device is not in grabbing. please call start_grabbing() first.')
            return
        from genicam.gentl import TimeoutException
        try:
            img = self._aquire_img(ia, timeout)
        except TimeoutException:
            logger.error(f'[{ia.device.id_}] get_one_frame time out.')
            return None
        return img

    def software_trigger(self, id_):
        '''
        software trigger device to capture one image.
        refer below to let device work in software trigger mode:
            cam.set_property(id_, "TriggerMode", "On")
            cam.get_property(id_, "TriggerSource")
        '''
        ia = self.connect_ias[id_]
        ia.remote_device.node_map.TriggerSoftware.execute()

    def dump_properties(self, filepath, id_=None):
        '''
        Dump device properties to a json file.
        filepath: json file path
        id_: device id, if not set. will dump properties for all the connected devices.
        '''
        assert len(self.connect_ias) > 0, 'No device connected!'
        if id_ is not None:
            assert id_ in self.connect_ias.keys(), f'{id_} is not connected!'
        export_ias = list(self.connect_ias.values()) if id_ is None else [self.connect_ias[id_]]

        dev_properties = {}
        for ia in export_ias:
            properties = self.get_all_properties(ia.device.id_)
            dev_properties[ia.device.id_] = properties

        import json
        json_data = json.dumps(dev_properties, indent=4, separators=(',', ': '), ensure_ascii=False)
        with open(filepath, 'w') as f:
            f.write(json_data)
        logger.info(f'Dump properties for devices: {list(dev_properties.keys())}')

    def load_properties(self, filepath, id_=None):
        '''
        Load device properties from a json file.
        filepath: json file path
        id_: device id, if not set. will load properties for
             all the connected devices from the json file if the id_ is match.
        '''
        assert len(self.connect_ias) > 0, 'No device connected!'
        if id_:
            assert id_ in self.connect_ias.keys(), f'{id_} is not connected!'
        with open(filepath, 'r') as f:
            import json
            dev_properties = json.loads(f.read())
            if id_ is None:
                for exp_id, properties in dev_properties.items():
                    if exp_id in self.connect_ias.keys():
                        logger.info(f'Load properties for dev:{exp_id}')
                        for k, v in properties.items():
                            self.set_property(exp_id, k, v)
                    else:
                        logger.warning(f'{exp_id} not connected, Ignore load properties for it!')
            else:
                if id_ not in dev_properties.keys():
                    logger.error(f'Load properties fail, {id_} not found in the json file!')
                else:
                    logger.info(f'Load properties for dev:{id_}')
                    for k, v in dev_properties[id_].items():
                        self.set_property(id_, k, v)

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
        from PIL import Image,ImageTk
        import tkinter as tk

        if not isinstance(id_list, list):
            ia_list = [self.connect_ias[id_list]]
        else:
            ia_list = [self.connect_ias[id_] for id_ in id_list]

        self.preview_list = ia_list
        self.preview_app = tk.Tk()
        self.preview_app.withdraw() #Hide main window

        for ia in ia_list:
            if ia.is_acquiring():
                logger.error(f'[{ia.device.id_}] Preview fail! Another grabbing task is running!')
                continue
            window_name = f'preview_{ia.device.id_}'
            win = tk.Toplevel()
            win.title(window_name)
            win.protocol("WM_DELETE_WINDOW", self._on_preview_close)
            ia.previewing = True
            ia.preview_win = win
            args = (ia, win_width, modify_func)
            self.preview_thread = threading.Thread(target=self._preview_func, daemon=True, args=args)
            self.preview_lock = threading.Lock()
            self.preview_thread.start()
            logger.info( f'[{ia.device.id_}] Start Preview')
        self.preview_app.mainloop()

    def _on_preview_close(self):
        for ia in self.preview_list:
            ia.previewing = False
            ia.preview_win.destroy()
        self.preview_app.destroy()
        logger.info('All preview stopped!')

    def _preview_func(self, ia, win_width, modify_func):
        from genicam.gentl import TimeoutException
        from PIL import Image,ImageTk
        import tkinter as tk

        if not ia.is_acquiring():
            ia.start_acquisition()

        ia.preview_label = None
        while ia.previewing:
            with self.preview_lock:
                try:
                    img = self._aquire_img(ia, 100)
                    if img is None:
                        continue
                    if modify_func:
                        img = modify_func(img)
                    preview_w = win_width if img.shape[0] >= win_width else img.shape[0]
                    preview_h = int(preview_w * (img.shape[1] / img.shape[0]))

                    image = Image.fromarray(img)
                    img_resized = image.resize((preview_h, preview_w), Image.CUBIC)
                    img_resized = ImageTk.PhotoImage(img_resized)

                    if ia.preview_label is None:
                        ia.preview_win.geometry(f'{preview_h}x{preview_w}')
                        ia.preview_label = tk.Label(ia.preview_win, image=img_resized)
                        ia.preview_label.grid()
                    else:
                        ia.preview_label.configure(image=img_resized)
                        ia.preview_label.imgtk = img_resized
                except TimeoutException:
                    logger.warning(f'[{ia.device.id_}] _preview_func Aquire image time out.')
                    break
                # except Exception as exc:
                #     print(exc)
                #     break
        ia.stop_acquisition()
        logger.info( f'[{ia.device.id_}] Preview thread Stopped!')

    def _grabbing_thread_func(self, ia, img_cbs):
        from genicam.gentl import TimeoutException

        while ia.is_acquiring():
            with ia.lock:
                try:
                    img = self._aquire_img(ia)
                    if img is None:
                        logger.warning(f'[{ia.device.id_}]aquire_img fail!')
                        continue
                    else:
                        if ia.device.id_ in img_cbs.keys():
                            for cb in img_cbs[ia.device.id_]:
                                cb(ia.device.id_, img)
                except TimeoutException:
                    logger.warning(f'[{ia.device.id_}] aquire_img timeout!')
        logger.info(f'[{ia.device.id_}] _grabbing_thread_func exit!')

    def _aquire_img(self, ia, timeout=10):
        with ia.fetch_buffer(timeout=timeout,cycle_s=0.01) as buffer:
            if len(buffer.payload.components) <= 0:
                logger.error(f'[{ia.device.id_}] payload buffer is empty.')
                return None
            component = buffer.payload.components[0]
            width = component.width
            height = component.height

            from harvesters.util.pfnc import mono_location_formats
            channel = int(component.num_components_per_pixel)
            if channel == 1:
                content = component.data.reshape(height, width)
            else:
                content = component.data.reshape(height, width, channel)
            img = content.copy()
            return img
