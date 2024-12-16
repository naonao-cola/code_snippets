'''
Copyright (C) 2023 TuringVision
'''
import os
import concurrent.futures
from threading import RLock
from .ftp_client import FtpClient

__all__ = ['MtFtpLoader']

class MtFtpLoader:

    def __init__(self, workers=4):
        self._workers = workers
        self._ftp_config = None
        self._lock = RLock()

    def set_ftp_config(self, **args):
        self._ftp_config = args

    def load_floder(self, ftp_dir, save_dir, callback=None):
        with FtpClient(**self._ftp_config) as ftp:
            file_list = ftp.get_dir_file_list(ftp_dir)
            if not file_list:
                return 0

        worker_file_list = [file_list[i::self._workers] for i in range(self._workers)]
        worker_percent = [0 for i in range(self._workers)]

        def _atomic_callback(worker_index, percent):
            with self._lock:
                worker_percent[worker_index] = percent
                return callback(int(sum(worker_percent)/self._workers))

        ftp_client_list = []
        for i in range(self._workers):
            ftp_client = FtpClient(**self._ftp_config)
            ftp_client_list.append(ftp_client)

        result = [None for i in range(self._workers)]
        with concurrent.futures.ThreadPoolExecutor(self._workers) as e:
            for i in range(self._workers):
                cb = lambda p, wi=i: _atomic_callback(wi, p) if callback else None
                result[i] = e.submit(ftp_client_list[i].load_list,
                                  worker_file_list[i], save_dir, ftp_dir, cb)

        for ftp_client in ftp_client_list:
            ftp_client.close()

        total_load = 0
        for future in result:
            if future:
                total_load += future.result()
        return total_load
