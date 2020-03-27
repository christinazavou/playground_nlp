
import logging
import os


def get_logger(name, level, folder=None, filename=None):
    logger = logging.getLogger(name)
    logger.setLevel(level)
    formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    if folder and filename:
        handler = logging.FileHandler(os.path.join(folder, filename))
        handler.setLevel(level)
        handler.setFormatter(formatter)
        logger.addHandler(handler)
    return logger


def log_info(message, logger=None):
    if logger:
        logger.info(message)
    else:
        print(message)
