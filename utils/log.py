import logging
import time
import os
import sys, shutil

def get_logger(cfg, config_path):
    logdir = f'run/{cfg["dataset"]}/{time.strftime("%Y-%m-%d-%H-%M-%S")}'
    os.makedirs(logdir)
    shutil.copy(config_path, logdir)

    # create log directory
    if not os.path.exists(logdir):
        os.makedirs(logdir)

    # create log file
    logname = f'run-{time.strftime("%Y-%m-%d-%H-%M")}.log'
    log_file = os.path.join(logdir, logname)

    # create log
    logger = logging.getLogger('trian')
    logger.setLevel(logging.INFO)

    # set log format
    formatter = logging.Formatter('%(asctime)s | %(message)s', datefmt='%Y-%m-%d %H:%M:%S')

    # StreamHandler: Output the log to console
    stream_handler = logging.StreamHandler(sys.stdout)
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    # FileHandler: Output the log to log_file
    file_handler = logging.FileHandler(log_file)
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)
    cfg['logdir'] = logdir
    logger.info(f"Conf | use logdir {logdir}")

    return logger