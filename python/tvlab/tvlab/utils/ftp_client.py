'''
Copyright (C) 2023 TuringVision
'''
import os
import logging
import time
from ftplib import FTP
from ftplib import error_perm

__all__ = ['FtpClient']

logger = logging.getLogger()


def _ftp_block_callback(file_size, percent_callback=None, process_block=None):
    """ load/upload block callback

    :param file_size(int): total file size
    :param percent_callback(function):
    :param process_block(function):
    :return: (function)
    """
    load_progress = [0, -1] # [load_size, load_percent]
    def cb_wrapper(data):
        if process_block:
            process_block(data)
        if percent_callback:
            load_progress[0] += len(data)
            percent = int(100 * load_progress[0] / file_size)
            if percent != load_progress[1]:
                load_progress[1] = percent
                percent_callback(percent)
    return cb_wrapper


class FtpClient:
    """The ftp client class to upload/load file to/from FTP server

    :param host: (str) The IP of the FTP server
    :param port: (int) The port of the FTP server
    :param username: (str) username of the FTP server
    :param password: (str) password of the FTP server
    """

    TIMEOUT = 5.0
    RETRY_TIMES = 3

    def __init__(self, host, port=21, username="anonymous", password="anonymous", pasv_mode=True):
        """

        :param host: (str) The IP of the FTP server
        :param port: (int) The port of the FTP server
        :param username: (str) username of the FTP server
        :param password: (str) password of the FTP server
        :param pasv_mode: (bool) enable passive mode
        """
        self._host = host
        self._port = port
        self._username = username
        self._password = password
        self._ftp = None
        self._pasv_model = pasv_mode
        self._connect()

    def _connect(self):
        """start ftp connect

        :return:
        """
        ftp = FTP(timeout=self.TIMEOUT)
        ftp.encoding='utf-8'
        try:
            ftp.connect(self._host, self._port, timeout=self.TIMEOUT)
            self._ftp = ftp
            self._ftp.login(self._username, self._password)
            self._ftp.set_pasv(self._pasv_model)
        except Exception as e:
            self.close()
            self._ftp = None

    def _reconnect(self):
        """reconnect ftp

        :return:
        """
        self.close()
        self._connect()

    def is_connected(self):
        return self._ftp is not None

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.close()

    def close(self):
        """close ftp connect

        :return:
        """
        if self._ftp:
            self._ftp.close()
            self._ftp = None

    def _safe_ftp_op(self, ftp_func):
        for retry in range(self.RETRY_TIMES):
            try:
                if self._ftp:
                    return ftp_func()
            except error_perm as e:
                logger.warning('ftp op exception: %s', str(e))
                return None
            except Exception as e:
                pass
            time.sleep(self.TIMEOUT ** (retry + 1))
            self._reconnect()
        return None

    def load(self, ftp_file_path, save_dir, percent_callback=None):
        """load file from ftp server

        :param ftp_file_path(str): ftp file path
        :param save_dir(str): local save directory
        :param percent_callback(function):
        :return: (bool)
        """

        os.makedirs(save_dir, exist_ok=True)

        target_name = os.path.join(save_dir, os.path.basename(ftp_file_path))

        file_size = self._safe_ftp_op(lambda: self._ftp.size(ftp_file_path))

        if not file_size:
            return False

        for retry in range(self.RETRY_TIMES):
            with open(target_name, 'wb') as pfile:
                try:
                    callback = _ftp_block_callback(file_size, percent_callback, pfile.write)
                    self._ftp.retrbinary("RETR " + ftp_file_path, callback)
                    return True
                except Exception as e:
                    logger.warning('ftp load exception: %s', str(e))
                    logger.warning("%s load Failed!, retry %s", ftp_file_path, retry)
                    time.sleep(self.TIMEOUT ** (retry + 1))
                    self._reconnect()
            if os.path.exists(target_name):
                os.remove(target_name)
        return False

    def load_list(self, file_list, save_dir, ftp_dir=None, percent_callback=None):
        """load file list from ftp server

        :param file_list(list): ftp file path list
        :param save_dir(str): local save directory
        :param ftp_dir(str): ftp dir, for keep same directory structure locally as FTP
        :param percent_callback(function): return True to abort this load task
        :return: (int) file number of success load
        """
        if self._ftp is None:
            return 0

        if ftp_dir and ftp_dir[-1] == '/':
            ftp_dir = ftp_dir[:-1]

        total_file_num = len(file_list)
        if total_file_num == 0:
            return 0

        file_set = set(file_list)
        last_cb_percent = -1
        load_file_num = 0
        for i, file in enumerate(file_set):
            percent = int(100 * i / total_file_num)
            if percent != last_cb_percent:
                last_cb_percent = percent
                if percent_callback and percent_callback(percent):
                    return 0
            local_dir = save_dir
            if ftp_dir:
                local_dir = os.path.join(save_dir, os.path.dirname(file[len(ftp_dir)+1:]))
            if self.load(file, local_dir):
                load_file_num += 1

        if percent_callback:
            percent_callback(100)
        return load_file_num

    def is_ftp_dir(self, ftp_file_path, guess_by_extension=True):
        """ simply determines if an item listed on the ftp server is a valid directory or not
            if the name has a "." in the fourth to last position, its probably a file extension
            this is MUCH faster than trying to set every file to a working directory,
            and will work 99% of time.

        :param ftp_file_path(str): ftp file path
        :param guess_by_extension(bool):
        :return: (bool)
        """
        if guess_by_extension is True:
            if ftp_file_path[-4] == '.' or ftp_file_path[-5] == '.':
                return False
            else:
                return True

        ret = False
        original_cwd = self._safe_ftp_op(lambda: self._ftp.pwd())
        if original_cwd:
            try:
                self._ftp.cwd(ftp_file_path)
                ret = True
            except Exception:
                pass
            self._safe_ftp_op(lambda: self._ftp.cwd(original_cwd))
        return ret

    def ls_dir(self, ftp_dir):
        file_list = self._safe_ftp_op(lambda: self._ftp.nlst(ftp_dir))
        if not file_list:
            return []
        return file_list

    def mk_dir(self, ftp_dir):
        if len(ftp_dir) <= 1:
            return True

        if not self.is_ftp_dir(ftp_dir, guess_by_extension=False):
            self.mk_dir(os.path.dirname(ftp_dir))
        else:
            return True

        ret = self._safe_ftp_op(lambda: self._ftp.mkd(ftp_dir))
        if not ret:
            return False
        return True

    def get_dir_file_list(self, ftp_dir):
        """get ftp directory file list

        :param file_dir(str): ftp directory
        :return: (list) ftp file list
        """
        file_list = []
        for file in self.ls_dir(ftp_dir):
            if self.is_ftp_dir(file):
                file_list += self.get_dir_file_list(file)
            else:
                file_list.append(file)
        return file_list

    def load_floder(self, ftp_dir, save_dir, percent_callback=None):
        """load directory from ftp

        :param file_dir(str): ftp directory
        :param save_dir(str): local save directory
        :param percent_callback(function): return True to abort this load task
        :return: (int) file number of success load
        """
        if self._ftp is None:
            return 0

        if not self.is_ftp_dir(ftp_dir):
            return 0

        file_list = self.get_dir_file_list(ftp_dir)
        if not file_list:
            return 0

        return self.load_list(file_list, save_dir, ftp_dir, percent_callback)

    def upload(self, local_file, ftp_file_path, percent_callback=None):
        """upload file to ftp

        :param local_file(str): local file path
        :param ftp_file_path(str): ftp file path
        :return: (bool)
        """
        if self._ftp is None:
            return False

        try:
            file_size = os.path.getsize(local_file)
        except Exception:
            return False

        if file_size == 0:
            return False

        for retry in range(self.RETRY_TIMES):
            try:
                with open(local_file, 'rb') as pfile:
                    self._ftp.storbinary("STOR " + ftp_file_path, pfile, callback=
                                         _ftp_block_callback(file_size, percent_callback))
                    return True
            except Exception as e:
                logger.warning('ftp upload exception: %s', str(e))
                logger.warning("%s upload Failed!, retry %s", ftp_file_path, retry)
                time.sleep(self.TIMEOUT ** (retry + 1))
                self._reconnect()
        return False
