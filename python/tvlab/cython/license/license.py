import time
import numpy as np
import os
import threading
try:
    import msvcrt
except ImportError:
    msvcrt = None

try:
    TimeoutError
except NameError:
    TimeoutError = OSError


class Timeout(TimeoutError):
    """
    Raised when the lock could not be acquired in *timeout*
    seconds.
    """

    def __init__(self, lock_file):
        """
        """
        #: The path of the file lock.
        self.lock_file = lock_file
        return None

    def __str__(self):
        temp = "The file lock '{}' could not be acquired."\
               .format(self.lock_file)
        return temp


class _Acquire_ReturnProxy(object):

    def __init__(self, lock):
        self.lock = lock
        return None

    def __enter__(self):
        return self.lock

    def __exit__(self, exc_type, exc_value, traceback):
        self.lock.release()
        return None


class BaseFileLock(object):
    """
    Implements the base class of a file lock.
    """

    def __init__(self, lock_file, timeout = -1):
        """
        """
        self._lock_file = lock_file

        self._lock_file_fd = None

        self.timeout = timeout

        self._thread_lock = threading.Lock()

        self._lock_counter = 0
        return None

    @property
    def lock_file(self):
        return self._lock_file

    @property
    def timeout(self):
        return self._timeout

    @timeout.setter
    def timeout(self, value):
        self._timeout = float(value)
        return None

    # Platform dependent locking
    # --------------------------------------------

    def _acquire(self):
        raise NotImplementedError()

    def _release(self):
        raise NotImplementedError()

    # Platform independent methods
    # --------------------------------------------

    @property
    def is_locked(self):
        return self._lock_file_fd is not None

    def acquire(self, timeout=None, poll_intervall=0.05):
        # Use the default timeout, if no timeout is provided.
        if timeout is None:
            timeout = self.timeout

        # Increment the number right at the beginning.
        # We can still undo it, if something fails.
        with self._thread_lock:
            self._lock_counter += 1

        lock_id = id(self)
        lock_filename = self._lock_file
        start_time = time.time()
        try:
            while True:
                with self._thread_lock:
                    if not self.is_locked:
                        self._acquire()

                if self.is_locked:
                    break
                elif timeout >= 0 and time.time() - start_time > timeout:
                    raise Timeout(self._lock_file)
                else:
                    time.sleep(poll_intervall)
        except:
            # Something did go wrong, so decrement the counter.
            with self._thread_lock:
                self._lock_counter = max(0, self._lock_counter - 1)

            raise
        return _Acquire_ReturnProxy(lock = self)

    def release(self, force = False):
        with self._thread_lock:

            if self.is_locked:
                self._lock_counter -= 1

                if self._lock_counter == 0 or force:
                    lock_id = id(self)
                    lock_filename = self._lock_file

                    self._release()
                    self._lock_counter = 0

        return None

    def __enter__(self):
        self.acquire()
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.release()
        return None

    def __del__(self):
        self.release(force = True)
        return None


class WindowsFileLock(BaseFileLock):
    """
    Uses the :func:`msvcrt.locking` function to hard lock the lock file on
    windows systems.
    """

    def _acquire(self):
        open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC

        try:
            fd = os.open(self._lock_file, open_mode)
        except OSError:
            pass
        else:
            try:
                msvcrt.locking(fd, msvcrt.LK_NBLCK, 1)
            except (IOError, OSError):
                os.close(fd)
            else:
                self._lock_file_fd = fd
        return None

    def _release(self):
        fd = self._lock_file_fd
        self._lock_file_fd = None
        msvcrt.locking(fd, msvcrt.LK_UNLCK, 1)
        os.close(fd)

        try:
            os.remove(self._lock_file)
        # Probably another instance of the application
        # that acquired the file lock.
        except OSError:
            pass
        return None


# Unix locking mechanism
# ~~~~~~~~~~~~~~~~~~~~~~

class UnixFileLock(BaseFileLock):
    """
    Uses the :func:`fcntl.flock` to hard lock the lock file on unix systems.
    """

    def _acquire(self):
        import fcntl
        open_mode = os.O_RDWR | os.O_CREAT | os.O_TRUNC
        fd = os.open(self._lock_file, open_mode)

        try:
            fcntl.flock(fd, fcntl.LOCK_EX | fcntl.LOCK_NB)
        except (IOError, OSError):
            os.close(fd)
        else:
            self._lock_file_fd = fd
        return None

    def _release(self):
        import fcntl
        fd = self._lock_file_fd
        self._lock_file_fd = None
        fcntl.flock(fd, fcntl.LOCK_UN)
        os.close(fd)
        return None

FileLock = None

if msvcrt:
    FileLock = WindowsFileLock
else:
    FileLock = UnixFileLock


def __obj_to_json(obj, json_path):
    import os, json
    dir_name = os.path.dirname(json_path)
    if dir_name:
        os.makedirs(dir_name, exist_ok=True)
    with open(json_path, 'wt') as fp:
        json.dump(obj, fp)


def __obj_from_json(json_path):
    import json
    with open(json_path, 'rt') as fp:
        return json.load(fp)


# add license_check
def hash_md5(info):
    import hashlib
    md5 = hashlib.md5()
    md5.update(info)
    return md5.hexdigest()


def hash_fg(fg, password):
    a = fg.encode()
    b = password.encode()
    return bytes([c+b[i%len(b)] for i, c in enumerate(a)])


def unhash_fg(fg, password):
    ac = fg
    b = password.encode()
    return bytes([c - b[i%len(b)] for i, c in enumerate(ac)]).decode()


def __read_uuid():
    import os, json
    import subprocess
    import platform
    system = platform.system()

    if system == 'Windows':
        import sys
        import pythoncom
        is_main_thread = threading.current_thread() is threading.main_thread()
        if not is_main_thread:
            pythoncom.CoInitialize()
        import wmi
        c = wmi.WMI()
        media = c.Win32_PhysicalMedia()
        uuid = media[0].SerialNumber
        for media_i in media:
            if 'PHYSICALDRIVE0' in media_i.tag:
                uuid = media_i.SerialNumber
        if not is_main_thread:
            pythoncom.CoUninitialize()
    elif system == 'Linux':
        cmd = 'blkid /dev/sda1'
        try:
            result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, check=True)
            result = result.stdout.strip().decode()
            disk_uuid = result[result.find('UUID'):].split()[0][6:-1]
        except Exception:
            disk_uuid = 'nosda1'

        if disk_uuid == 'nosda1':
            cmd = 'lsblk -n -a -o UUID /dev/sda1'
            try:
                result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, check=True)
                disk_uuid = result.stdout.strip().decode()
            except Exception:
                disk_uuid = 'nosda1'

        try:
            with open('/etc/machine-id', 'rt') as fp:
                machine_id = fp.read()[:-1]
        except Exception:
            machine_id = 'nomachine_id'
        uuid = disk_uuid + machine_id
    elif system == 'Darwin':
        cmd = "system_profiler SPHardwareDataType | awk '/UUID/ { print $3; }'"
        result = subprocess.run(cmd, stdout=subprocess.PIPE, shell=True, check=True)
        uuid = result.stdout.strip().decode()
    return uuid


def get_fingerprint(passwd):
    import rsa
    api_priv_key_str = '''
    -----BEGIN RSA PRIVATE KEY-----
    MIICYwIBAAKBgQCPmW5zME7Y5ysuJmDs82PlJQ1NGbthivgUHk6dN7wb4bCHEJN5
    GgkF9dd8VCuQIIv5uc4PUfxKlvAJY1amOTcPCXPkxJopyGSx9IE8AUT9NP5SKWow
    ANXTgplwRE9nmebXkN3Fr520yklg8iZrb5t/WANMgEVsJd3PAj1BjtTUIwIDAQAB
    AoGBAIkVPIZEEJEppWQKCS9Kbgua4ma+8M1+X7K89+lIApBPVDirz0ylWQXSmaI7
    q9aL63Q7NwYTCxidnIopxSiX4MyIMa4cVJ9dmIQiVGcBos/idHwjgayOsq7qYxHH
    3eBPz/OTKmEleINJYkfmhYRkYga9CIRNWTtJmJwI2Bnbb60BAkUAtt+JMoaaEtsS
    el2nOvUQ5vNgyWPj1s4UMY2kMF94hOvz2sAJMhYXD6Vmb0J4OJhwoAvG5djqFiGm
    NdaivkdouORd5aMCPQDJBX0Yw4ybNrZ9XP8Q41kFSzELLNpkN7uHm1Agy+0nCm5t
    ZVVAW3VxcsCOA6XNcAqC4J0ynwkCT5YLP4ECRQCDsHGCGv/0uCGUUMDOag/j4DtA
    i1hVJU3RaWhlFTsarTeLvWJh2Rp+P+OGF12vk8b22cQ/XHclvNGOT6QeVhoJmxoq
    9QI9AMO0sEG3v+AsuCX8r+aFMFnTBoBWvpfrGou/TZdgchYcNb4TdZgecoxsp8Kb
    EoSbm+AnRWPPKToyuWViAQJFAIgYxDKVmMunuWiHK4fVSYVHMAEvBlVb0Bn0EyCi
    bTCiKmVNJW6gtfXjAH5BtWF7ExxKv4vkvGSnBmPoajOiUV0ctYE5
    -----END RSA PRIVATE KEY-----
    '''
    privkey = rsa.PrivateKey.load_pkcs1(api_priv_key_str.encode())

    passwd = rsa.decrypt(passwd, privkey).decode()

    if hasattr(get_fingerprint, '__cache_id__'):
        return hash_fg(get_fingerprint.__cache_id__, passwd)

    uuid = __read_uuid()

    uuid_bytes = uuid.encode('utf-8')
    uuid = hash_md5(uuid_bytes)
    hash_uuid =  hash_fg(uuid, passwd)
    get_fingerprint.__cache_id__ = uuid
    return hash_uuid


def save_fingerprint(out_path):
    import time
    import platform
    import base64

    uuid = __read_uuid()
    hash_uuid = hash_fg(uuid, 'tvlab@turingvision.com')
    fingerprint = base64.b64encode(hash_uuid).decode()
    cur_date = time.strftime('%Y%m%d')
    fg_info = {'fingerprint': fingerprint,
               'date': cur_date,
               'platform': platform.platform(),
               'pyver': platform.python_version()}
    __obj_to_json(fg_info, out_path)
    return fg_info


def show_fingerprint(fg_path, password):
    import base64
    fg_info = __obj_from_json(fg_path)
    fingerprint = fg_info['fingerprint']
    uuid = base64.b64decode(fingerprint.encode())
    uuid = unhash_fg(uuid, password)
    return uuid


def __load_key(path_or_str, is_public=True):
    import rsa
    import os
    key_str = path_or_str
    if os.path.isfile(path_or_str):
        with open(path_or_str,'r') as f:
            key_str = f.read()
    if is_public:
        key = rsa.PublicKey.load_pkcs1(key_str.encode())
    else:
        key = rsa.PrivateKey.load_pkcs1(key_str.encode())
    return key


def signature(fingerprint, privkey, date, package):
    import rsa
    import base64
    message = fingerprint + ',' + date + ',' + package
    sign = rsa.sign(message.encode(), privkey, 'SHA-1')
    sign = base64.b64encode(sign)
    sign_info = {'sign': sign.decode(), 'date': date, 'package': package}
    return sign_info

def verify_signature(fingerprint, sign_info, pubkey, package):
    import rsa
    import time
    import base64

    try:
        date = sign_info['date']
        message = fingerprint + ',' + date + ',' + package
        sign = sign_info['sign'].encode()
        sign = base64.b64decode(sign)
        ret = rsa.verify(message.encode(), sign, pubkey)
        cur_date = time.strftime('%Y%m%d')
        if int(cur_date) > int(date):
            return False
    except Exception:
        return False
    return True


def gen_license(fg_path, privkey_path, date, package, out_path):
    import time
    import base64
    fg_info = __obj_from_json(fg_path)
    fingerprint = fg_info['fingerprint']
    uuid = base64.b64decode(fingerprint.encode())
    uuid = unhash_fg(uuid, 'tvlab@turingvision.com')
    fingerprint = hash_md5(uuid.encode('utf-8'))
    #print('fingerprint:', fingerprint)
    privkey = __load_key(privkey_path, is_public=False)
    sign_info = signature(fingerprint, privkey, date, package)
    sign_info['cap-date'] = fg_info['date']
    sign_info['gen-date'] = time.strftime('%Y%m%d')
    __obj_to_json(sign_info, out_path)
    return sign_info


class ET199:
    def __init__(self, user_pin, debug=False):
        '''
        user_pin (bytes): 8bytes b'asdfasdf'
        '''
        from .c_et199 import CET199Dongle
        assert len(user_pin) == 8

        self._c = CET199Dongle(debug)
        self.user_pin = np.frombuffer(user_pin, np.uint8)

    def get_cnt(self):
        return self._c.get_cnt()

    def open(self, i):
        ret = self._c.open(i)
        return ret

    def close(self, i):
        return self._c.close(i)

    def get_hardware_id(self, i):
        '''
        bid (bytes): 8bytes
        '''
        bid = np.zeros(8, np.uint8)
        self._c.get_hardware_id(i, bid)
        return bid.tobytes()

    def write_sign(self, i, sign, date):
        '''
        sign (bytes): 128bytes b'asdfaxxxasdf'
        date (bytes): 8bytes b'20200309'
        '''
        assert len(sign) == 128
        assert len(date) == 8
        merge_d = np.frombuffer(sign + date, np.uint8)
        return self._c.write_sign(i, self.user_pin, merge_d)

    def read_sign(self, i):
        '''
        sign (bytes): 128bytes b'asdfaxxxasdf'
        date (bytes): 8bytes b'20200309'
        '''
        merge_d = np.zeros(128+8, np.uint8)
        self._c.read_sign(i, self.user_pin, merge_d)
        merge_d = merge_d.tobytes()
        sign = merge_d[:128]
        date = merge_d[-8:]
        return  sign, date

    def set_atr(self, i, atr):
        '''
        atr (bytes): 16bytes b'xxxxxxxxxxxxxx'
        '''
        assert len(atr) == 16
        atr = np.frombuffer(atr, np.uint8)
        return self._c.set_atr(i, atr)


    def get_atr(self, i):
        '''
        atr (bytes): 16bytes b'xxxxxxxxxxxxxx'
        '''
        atr = np.zeros(16, np.uint8)
        self._c.get_atr(i, atr)
        return atr.tobytes()


def make_license_dongle(private_key_path, user_pin, date, package, filter_func=None):
    '''
    private_key_path(str):
    user_pin (bytes): 8bytes
    date (str):
    package (str):
    '''
    import rsa
    import time, base64
    dog = ET199(user_pin, debug=True)
    for i in range(dog.get_cnt()):
        print('====== make dongle start ======')
        dog.open(i)
        bid = dog.get_hardware_id(i)
        bid = base64.b64encode(bid).decode('utf-8')
        atr = dog.get_atr(i).decode('utf-8')
        if filter_func and not filter_func(atr):
            print('====== skip make dongle ======')
            dog.close(i)
            continue
        message = bid + ',' + date + ',' + package
        privkey = __load_key(private_key_path, False)
        sign = rsa.sign(message.encode(), privkey, 'SHA-1')
        dog.write_sign(i, sign, date.encode('utf-8'))
        cur_date = time.strftime('%Y%m%d')
        atr = package.ljust(7) + cur_date + ' '
        dog.set_atr(i, atr.encode('utf-8'))
        dog.close(i)
        print('====== make dongle done ======')


def verify_license_dongle(user_pin, verify_func, filter_func=None):
    import platform
    import time, base64

    system = platform.system()

    # try verify from local cache
    if hasattr(verify_license_dongle, '__cache_id__'):
        [bid, sign, date, last_date] = verify_license_dongle.__cache_id__
        verify_func(bid, sign, date, last_date)
        return

    verify_license_dongle.__cache_id__ = [None, None, None, None]

    if system == 'Windows':
        import tempfile
        import os.path as osp
        temp_dir = tempfile.gettempdir()
        cache_path = osp.join(temp_dir, 'tvlab_verify_dongle_cache.json')
        cache_lock_path = osp.join(temp_dir, 'tvlab_verify_dongle_cache.json.lock')
    else:
        cache_path = '/dev/shm/tvlab_verify_dongle_cache.json'
        cache_lock_path = '/dev/shm/tvlab_verify_dongle_cache.json.lock'

    lock = FileLock(cache_lock_path, timeout=10)
    try:
        with lock.acquire():
            # try verify from global cache
            try:
                cache_info = __obj_from_json(cache_path)
                cur_date = time.strftime('%Y%m%d')
                sign_hash = cache_info['sign']
                sign_hash = base64.b64decode(sign_hash.encode())
                sign_hash = unhash_fg(sign_hash, cur_date)
                sign = base64.b64decode(sign_hash.encode())
                bid = cache_info['bid']
                date = cache_info['date']
                last_date = cache_info['last_date']
                ret = verify_func(bid, sign, date, last_date)
                if ret:
                    verify_license_dongle.__cache_id__ = [bid, sign, date, last_date]
                    return
            except Exception as e:
                pass

            # try verify from usb dongle
            verify_ok = False
            dog = ET199(user_pin, debug=False)
            for i in range(dog.get_cnt()):
                ret = dog.open(i)
                if ret != 0:
                    continue
                try:
                    bid = dog.get_hardware_id(i)
                    bid = base64.b64encode(bid).decode('utf-8')
                    atr = dog.get_atr(i).decode('utf-8')
                    if filter_func and not filter_func(atr):
                        dog.close(i)
                        continue
                    sign, date = dog.read_sign(i)
                    cur_date = time.strftime('%Y%m%d')
                    last_date = atr[-9:-1]
                    date = date.decode('utf-8')
                    ret = verify_func(bid, sign, date, last_date)
                    if ret:
                        verify_ok = True
                        verify_license_dongle.__cache_id__ = [bid, sign, date, last_date]
                        cur_date = time.strftime('%Y%m%d')
                        sign_hash = base64.b64encode(sign).decode()
                        sign_hash = hash_fg(sign_hash, cur_date)
                        sign_hash = base64.b64encode(sign_hash).decode()
                        cache_info = {'bid': bid, 'sign': sign_hash, 'date': date, 'last_date': last_date}
                        __obj_to_json(cache_info, cache_path)
                        atr = atr[:7] + cur_date + ' '
                        dog.set_atr(i, atr.encode('utf-8'))
                except Exception as e:
                    print(e)
                    pass
                dog.close(i)
                if verify_ok:
                    break
    except Timeout:
        print("Acquire global cahe lock timeout!")


def verify_tvlab_license_dongle():
    pubkey_str = '''
    -----BEGIN RSA PUBLIC KEY-----
    MIGJAoGBAL7t6/3OWg+L1P94CBWhK8njGEyWjfDDvU/N+2CGJvHxkakkdnAoIuv9
    ZddrBPcSWKONz0dGx9DG/9301/i+0o/0eYvbTjJR5j+24xuU1aTg/jzRwfBikGMF
    P86Qjxc9Sjd7hGi4WG3eZfl9655JcYuIPuKJtXh6vqR7HZZQpvULAgMBAAE=
    -----END RSA PUBLIC KEY-----
    '''

    def dongle_verify_func(bid, sign, date, last_date):
        import time
        import rsa

        try:
            package = 'tvlab'
            message = bid + ',' + date + ',' + package
            pubkey = rsa.PublicKey.load_pkcs1(pubkey_str.encode())
            ret = rsa.verify(message.encode(), sign, pubkey)
            cur_date = time.strftime('%Y%m%d')
            if int(last_date) <= int(cur_date) and int(cur_date) <= int(date):
                print('Verify OK!')
                return True
        except Exception as e:
            print(e)

        print('Verify Failed!')
        return False

    verify_license_dongle(b'tvindsai', dongle_verify_func, lambda x: x[:5] == 'tvlab')


def make_tvlab_master_license_dongle(privkey, date):
    ask_f = lambda x: input('>>> y/n:') == 'y'
    make_license_dongle(privkey, b'tvindsma', date, 'tvmalt', ask_f)


def verify_tvlab_master_license_dongle():
    pubkey_str = '''
    -----BEGIN RSA PUBLIC KEY-----
    MIGJAoGBAL7t6/3OWg+L1P94CBWhK8njGEyWjfDDvU/N+2CGJvHxkakkdnAoIuv9
    ZddrBPcSWKONz0dGx9DG/9301/i+0o/0eYvbTjJR5j+24xuU1aTg/jzRwfBikGMF
    P86Qjxc9Sjd7hGi4WG3eZfl9655JcYuIPuKJtXh6vqR7HZZQpvULAgMBAAE=
    -----END RSA PUBLIC KEY-----
    '''

    g_verify_ok = {'result': False}
    def dongle_verify_func(bid, sign, date, last_date):
        import time
        import rsa

        try:
            package = 'tvmalt'
            message = bid + ',' + date + ',' + package
            pubkey = rsa.PublicKey.load_pkcs1(pubkey_str.encode())
            ret = rsa.verify(message.encode(), sign, pubkey)
            cur_date = time.strftime('%Y%m%d')
            if int(last_date) <= int(cur_date) and int(cur_date) <= int(date):
                g_verify_ok['result'] = True
                print('Verify Master OK!')
                return True
        except Exception as e:
            print(e)

        print('Verify Master Failed!')
        return False

    verify_license_dongle(b'tvindsma', dongle_verify_func, lambda x: x.startswith('tvmalt'))
    return g_verify_ok['result']


def make_tvlab_license_dongle(date):
    if not verify_tvlab_master_license_dongle():
        print('master verify failed')
        return
    privkey_str = """
    -----BEGIN RSA PRIVATE KEY-----
    MIICYQIBAAKBgQC+7ev9zloPi9T/eAgVoSvJ4xhMlo3ww71Pzftghibx8ZGpJHZw
    KCLr/WXXawT3Elijjc9HRsfQxv/d9Nf4vtKP9HmL204yUeY/tuMblNWk4P480cHw
    YpBjBT/OkI8XPUo3e4RouFht3mX5feueSXGLiD7iibV4er6kex2WUKb1CwIDAQAB
    AoGBAJfGSD/lRoBvNu2x/DM9gLKnHQzc7Y6D+zyyUG7llZXk41aizqfPsBsKv2dk
    Anlpkx7Ivwo5AOQ9HO7TEAavgAkbGzFG3tpDqpAibU0H0Z0811WO0dY52n0zTikw
    2nA1PB3KFUZrBl+XDBEGWZylAi8oQ1guex16Oghea0LvOOXpAkUA/UN8FtnYR5qm
    g+JpT2wSI2dpVW0PNP82F6kw10lp+NlpDArHgh+tOD5DJ79kOZrDpLha7z2i5u29
    HHIHbkGGhi7JvOcCPQDA/gYIDWwFuRKJMa1f4GW7Dygys4NKBZTuEa9uMBUrHJBy
    hGropiGocHNrQbiQv1nQDkoVOgu3NALjPj0CRHT2B4aeEG3xE9lwZGYTaMUE6vZS
    qWU2P4rpze5+rvwHm6W+DKkha+O/jU/76ZNYz+VwZj56XpL7VAyg9KGMjU3GqdAd
    Aj0AiEsdYVRxRD1mjMMriLlFvuw+XEgQYRc+DT8qIGwOlwquLGG9yp2AyZ7YtBeO
    y5KHT/q3cPQ0T+aIX4mVAkRM+GxoVc8wqJ4gjec1vR6HGHmzHU2+zlLMTt5i/Kjr
    qzN7EF9SZjScgmbANlEF7TKUVvrSjweFC6iN+Zdr77kznS5Z/g==
    -----END RSA PRIVATE KEY-----
    """
    def filter_master(x):
        print(x)
        return not x.startswith('tvmalt')
    make_license_dongle(privkey_str, b'tvindsai', date, 'tvlab', filter_master)
