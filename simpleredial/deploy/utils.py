from functools import wraps
import logging
from pprint import pformat
import time

def timethis(func):
    '''
    Decorator that reports the execution time.
    '''
    @wraps(func)
    def wrapper(*args, **kwargs):
        start = time.time()
        result = func(*args, **kwargs)
        end = time.time()
        # print(func.__name__, end-start)
        return result, end-start
    return wrapper


def init_logging(args, pipeline=False):
    if pipeline:
        name = f'{args["dataset"]}_{args["recall"]["model"]}_{args["rerank"]["model"]}'
        path = f'{args["root_dir"]}/log/{args["dataset"]}/pipeline/{name}.log'
    else:
        name = f'{args["dataset"]}_{args["model"]}'
        path = f'{args["root_dir"]}/log/{args["dataset"]}/{args["model"]}/{name}.log'
    formatter = logging.Formatter('%(asctime)s : %(message)s', "%Y-%m-%d %H:%M:%S")
    fileHandler = logging.FileHandler(path, mode='a')
    fileHandler.setFormatter(formatter)
    vlog = logging.getLogger(name)
    vlog.setLevel(logging.INFO)
    vlog.addHandler(fileHandler) 
    print(f'[!] init the logging information over, save the information into the log file:')
    print(f'[!] - {name}: {path}')
    return vlog


def push_to_log(information, vlog):
    information = pformat(information)
    vlog.info(information)
