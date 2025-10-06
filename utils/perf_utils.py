import time
import os
from contextlib import contextmanager

import torch


_perf_stats = None
_step = 0


CUDA_SYNC = os.environ.get("CUDA_SYNC", "0") == "1"

if CUDA_SYNC:
    print("CUDA_SYNC is enabled.")

def flatten_dict(d, parent_key='', sep='/'):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, dict):
            items.extend(flatten_dict(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def reset_perf_stats():
    global _perf_stats
    _perf_stats = {
        "time": {},
        "mem_gb_before": {},
        "mem_gb_after": {},
    }
reset_perf_stats()


@contextmanager
def measure_time_mem_context(tag):
    torch.cuda.reset_peak_memory_stats()

    _perf_stats["mem_gb_before"][tag] = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3

    if CUDA_SYNC:
        torch.cuda.synchronize()
    start = time.time()
    yield
    if CUDA_SYNC:
        torch.cuda.synchronize()
    end = time.time()
    
    _perf_stats["mem_gb_after"][tag] = torch.cuda.memory_stats()["allocated_bytes.all.peak"] / 1024 ** 3

    _perf_stats["time"][tag] = end - start


@contextmanager
def measure_time_context(tag):
    if CUDA_SYNC:
        torch.cuda.synchronize()
    start = time.time()
    yield
    if CUDA_SYNC:
        torch.cuda.synchronize()
    end = time.time()

    _perf_stats["time"][tag] = end - start


def measure_time(tag):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with measure_time_context(tag):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def measure_time_mem(tag):
    def decorator(func):
        def wrapper(*args, **kwargs):
            with measure_time_mem_context(tag):
                return func(*args, **kwargs)
        return wrapper
    return decorator


def pop_perf_stats(flatten=False):
    global _perf_stats, _step
    res = _perf_stats
    reset_perf_stats()

    if flatten:
        res = flatten_dict(res, sep="/")

    # res["step"] = _step
    _step += 1

    return res