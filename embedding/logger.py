# -*- coding: utf-8 -*-

import logging
from logging.handlers import RotatingFileHandler


class Log:
    formatter = logging.Formatter(
        '%(asctime)s %(filename)s[line:%(lineno)d] %(levelname)s %(thread)d %(threadName)s %(message)s')
    log_file_handler = RotatingFileHandler(filename='log/sentiment', maxBytes=10000000, mode='a', backupCount=7)
    log_file_handler.setFormatter(formatter)
    logging.basicConfig(level=logging.INFO)
    log = logging.getLogger()
    log.addHandler(log_file_handler)

    @staticmethod
    def debug(msg, *args, **kwargs):
        Log.log.debug(msg, *args, **kwargs)

    @staticmethod
    def info(msg, *args, **kwargs):
        Log.log.info(msg, *args, **kwargs)

    @staticmethod
    def warning(msg, *args, **kwargs):
        Log.log.warning(msg, *args, **kwargs)

    @staticmethod
    def error(msg, *args, **kwargs):
        Log.log.error(msg, *args, **kwargs)

    @staticmethod
    def exception(msg, *args, **kwargs):
        Log.log.exception(msg, *args, **kwargs)
