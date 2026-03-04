


from contextlib import contextmanager
from pathlib import Path
import sys

import yaml


def seed_everything(seed: int = 42):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


@contextmanager
def insert_import_path(path: str):
    sys.path.insert(0, path)
    utils_module = sys.modules.pop("utils", None)
    model_module = sys.modules.pop("model", None)

    yield

    if model_module is not None:
        sys.modules["model"] = model_module
    
    if utils_module is not None:
        sys.modules["utils"] = utils_module
    sys.path.pop(0)


def get_missing_run_ids(cfg_file):
    with open(cfg_file, "r") as f:
        cfgs = yaml.safe_load(f)

    if not isinstance(cfgs, list):
        return None

    all_ids = range(len(cfgs))

    result_dir = cfg_file.replace("cfg/", "results/").replace(".yaml", "")
    results = Path(result_dir).glob("*.json.gz")

    existing_ids = []
    for r in results:
        try:
            existing_ids.append(int(r.stem.split(".")[0]))
        except ValueError:
            pass

    missing_ids = sorted(set(all_ids) - set(existing_ids))
    return missing_ids
