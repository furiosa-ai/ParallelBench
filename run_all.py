import argparse
import itertools
from math import ceil
from multiprocessing import Process, Queue
import multiprocessing
import os
from pathlib import Path
import sys
from queue import Empty
import subprocess
from threading import Thread
import time
import importlib

from tqdm import tqdm
import yaml

from utils.utils import get_missing_run_ids


def check_gpu_free(dev_id):
    if "," in dev_id:
        dev_ids = [d.strip() for d in dev_id.split(",")]
        return all(check_gpu_free(d) for d in dev_ids)

    cmd = ["nvidia-smi", "-i", str(dev_id), "--query-gpu=utilization.gpu,memory.used", "--format=csv,noheader,nounits"]

    out = subprocess.check_output(cmd).decode("utf-8")
    lines = out.split("\n")

    device_util, used_mem_mb = [int(v) for v in lines[0].split(", ")]

    res = device_util == 0 and used_mem_mb < 1000
    return res


def wait_gpu_free(dev_id, free_count_req):
    if free_count_req is None:
        return

    free_count = 0
    sleep_sec = 30.0

    print(f"Waiting for GPU {dev_id} ({free_count_req}) to go idle...")
    while True:
        if check_gpu_free(dev_id):
            free_count += 1
            print(f"GPU {dev_id} free at {free_count}/{free_count_req}")

            if free_count >= free_count_req:
                return
        else:
            free_count = 0

        time.sleep(sleep_sec)


def iter_qu(qu: Queue, track_progress=False):
    if track_progress:
        total = qu.qsize()
        pbar = tqdm(total=total)

    while True:
        try:
            proc = qu.get(block=False)
            if track_progress:
                pbar.update(total - qu.qsize() - pbar.n)
        except Empty:
            break
        yield proc

    if track_progress:
        pbar.close()


def worker_main(worker_idx, script, qu, device, args=None):
    os.environ["PYTHONPATH"] = "."
    os.environ["CUDA_VISIBLE_DEVICES"] = str(device)

    script_module = importlib.import_module(script.replace("/", ".").replace(".py", ""))
    print(f"Running on device {device}")
    script_module.main(iter_qu(qu, track_progress=worker_idx == 0), {} if args is None else args)


def worker_func(worker_idx, script, qu: Queue, device, wait_free, args):
    wait_gpu_free(device, wait_free)

    n_retry = 1000

    for i in range(n_retry):
        try:
            p = Process(target=worker_main, args=(worker_idx, script, qu, device, args))
            p.start()
            p.join()
        except Exception as e:
            print(f"Error: {e}")
            print(f"Retrying...")

        if qu.empty():
            break


def load_cfg_list(arg):
    if ":" in arg:
        arg, run_ids = arg.split(":", 1)
        run_ids = [int(i) for i in run_ids.split(",")]
    else:
        run_ids = None

    if arg.endswith("_arglist.txt") or "/arglist/" in arg:
        assert run_ids is None
        with open(arg, "r") as f:
            return [l.strip() for l in f.readlines()]
    elif arg.endswith("_list.yaml"):
        with open(arg, "r") as f:
            cfgs = yaml.safe_load(f)

        extract_dir = Path("temp/") / arg.replace(".yaml", "")
        extract_dir.mkdir(parents=True, exist_ok=True)

        filenames = []
        for i, cfg in enumerate(cfgs):
            filename = extract_dir / f"{i:05d}.yaml"
            with open(filename, "w") as f:
                yaml.safe_dump(cfg, f, sort_keys=False)
            filenames.append(str(filename))

        print(len(filenames))
        if run_ids is not None:
            filenames = [filenames[i] for i in run_ids]

        return filenames
    else:
        assert run_ids is None
        return [arg]


def get_block_cfgs(cfg_files, block, blockwise):
    ind, size = block.split("/")
    ind = int(ind) - 1
    size = int(size)

    if not blockwise:
        return cfg_files[ind::size]
    else:
        cfgs_per_block = ceil(len(cfg_files) / size)
        start = ind * cfgs_per_block
        end = min(start + cfgs_per_block, len(cfg_files))
        return cfg_files[start:end]


def main():
    multiprocessing.set_start_method('spawn')
    os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

    parser = argparse.ArgumentParser()
    parser.add_argument("--device", type=str, required=True, nargs="+")
    parser.add_argument("--cfg", type=str, required=True, nargs="+")
    parser.add_argument("--skip_local", action="store_true")
    parser.add_argument("--wait_gpu_free", nargs="*", type=int)
    parser.add_argument("--confirm", action="store_true")
    parser.add_argument("--reverse", action="store_true")
    parser.add_argument("--block", nargs="*", action=None)
    parser.add_argument("--blockwise", action="store_true")
    parser.add_argument("--compute_metrics", action="store_true")
    parser.add_argument("--update_decoding_order", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--logger", type=str, default="wandb")
    parser.add_argument("script")
    args = parser.parse_args()

    # return args.cfg, vars(args)
    devices = args.device
    cfg_files = args.cfg
    script = args.script

    if args.skip_local:
        for i, cfg_file in enumerate(cfg_files):
            if ":" in cfg_file:
                cfg_file, sel_ind = cfg_file.split(":")
                sel_ind = list(set(list(map(int, sel_ind.split(",")))))
            else:
                sel_ind = None

            missing_ids = get_missing_run_ids(cfg_file)

            if sel_ind is not None:
                missing_ids = [i for i in missing_ids if i in sel_ind]

            if missing_ids is not None:
                cfg_files[i] = cfg_file + ":" + ",".join([str(i) for i in missing_ids])

    # if len(cfg_files) == 1:
    #     cfg_files = load_cfg_list(cfg_files[0])
    cfg_files = list(itertools.chain.from_iterable([load_cfg_list(arg) for arg in cfg_files]))

    if args.block is not None and len(args.block) > 0:
        cfg_files = [get_block_cfgs(cfg_files, block, args.blockwise) for block in args.block]

        if args.blockwise:
            cfg_files = list(itertools.chain.from_iterable(cfg_files))
        else:
            cfg_files = [x for x in itertools.chain.from_iterable(itertools.zip_longest(*cfg_files)) if x is not None]

    print(f"Running {len(cfg_files)} cfg files on devices {devices}")

    if args.reverse:
        cfg_files = cfg_files[::-1]

    if args.confirm:
        print("Press Enter to continue or Ctrl+C to cancel...")
        try:
            input()
        except KeyboardInterrupt:
            print("Cancelled")
            sys.exit(0)

    qu = Queue()

    for cfg in cfg_files:
        qu.put(cfg)

    func = worker_func

    if args.wait_gpu_free is not None:
        wait_gpu_free = 5

        if len(args.wait_gpu_free) > 0:
            wait_gpu_free = args.wait_gpu_free[0]
    else:
        wait_gpu_free = None

    func_args = {"no_progress": True}

    if args.compute_metrics:
        func_args["compute_metrics"] = True

    if args.update_decoding_order:
        func_args["update_decoding_order"] = True

    if args.logger is not None:
        func_args["logger"] = args.logger

    if args.skip_metrics is not None:
        func_args["skip_metrics"] = args.skip_metrics

    if args.overwrite is not None:
        func_args["overwrite"] = args.overwrite

    workers = [Thread(target=func, args=(worker_idx, script, qu, device, wait_gpu_free, func_args)) for worker_idx, device in enumerate(devices)]

    for worker in workers:
        worker.start()

    for worker in workers:
        worker.join()


if __name__ == "__main__":
    main()
