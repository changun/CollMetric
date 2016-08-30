import pyximport

pyximport.install()
import fast_utils
import numpy
from multiprocessing import Pool
import sys
import signal
import gc
import time
import random

_cur_sampler = None
_cur_sampler_signature = None


def sigterm_handler(_signo, _stack_frame):
    # Raises SystemExit(0):
    sys.exit(0)


def _init():
    # it is important to reseed the random generator in each process
    numpy.random.seed()
    signal.signal(signal.SIGINT, signal.SIG_IGN)
    signal.signal(signal.SIGHUP, sigterm_handler)
    signal.signal(signal.SIGTERM, sigterm_handler)
    signal.signal(signal.SIGUSR1, sigterm_handler)
    return True


def _init_sampler(train_dict, n_items, warp_count, batch_size):
    global _cur_sampler, _cur_sampler_signature
    # compute signature of the given arguments
    signature = (sum(map(lambda u: sum(u[1]) % (u[0] + 1), train_dict.items())),
                 n_items,
                 warp_count,
                 batch_size,)
    # only recreate sampler if the signature are different
    if _cur_sampler_signature != signature:
        _cur_sampler_signature = signature
        _cur_sampler = fast_utils.sample_warp(train_dict, n_items, warp_count, batch_size)
    time.sleep(1)


def _sample():
    return _cur_sampler.next()


# def _del():
#     global _cur_sampler
#     _cur_sampler = None
#     gc.collect()
#     time.sleep(1)
#     print ("Clear sampler")


_pool = Pool(processes=8, initializer=_init)
sampler_cache = {}


def create_sampler(train_dict, n_items, warp_count, batch_size):
    try:
        global _pool
        async_samples = []
        for _ in range(_pool._processes):
            _pool.apply_async(_init_sampler, (train_dict, n_items, warp_count, batch_size))
        for _ in range(_pool._processes):
            async_samples.append(_pool.apply_async(_sample))
        while True:
            async_sample = async_samples.pop(0)
            async_samples.append(_pool.apply_async(_sample))
            yield async_sample.get()
    finally:
        # for _ in range(_pool._processes):
        #     _pool.apply_async(_pool.apply_async(_del))
        pass
