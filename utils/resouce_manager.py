


import functools
import gc

import torch
import json

from dataset import load_dataset
from model import load_model
from transformers import AutoTokenizer


class ResourceManager:
    def __init__(self):
        self.resources = {}

    def config_to_key(self, args):
        return json.dumps(args, sort_keys=True)

    def load_resource(self, type, config, load_func):
        key = self.config_to_key(config)

        if type not in self.resources:
            self.resources[type] = {}

        if key not in self.resources[type]:
            if len(self.resources[type]) > 0:
                print(f"Clearing resources of type {type} before loading new resource.")
                self.resources[type] = {}
                gc.collect()
                torch.cuda.empty_cache()

            print(f"Loading resource {type} with {config}")
            self.resources[type][key] = load_func(**config)

        return self.resources[type][key]

    def load_model(self, **kwargs):
        return self.load_resource("model", kwargs, load_model)

    def load_dataset(self, **kwargs):
        return self.load_resource("dataset", kwargs, load_dataset)
    
    def load_tokenizer(self, **kwargs):
        return self.load_resource("tokenizer", kwargs, functools.partial(
            AutoTokenizer.from_pretrained, trust_remote_code=True))
