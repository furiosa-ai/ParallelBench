# seed
# no_grad


import argparse
import functools
import json
import os
from pathlib import Path
import time
import torch
from tqdm import tqdm
import yaml
import pandas as pd
import gzip

from dataset import load_dataset
from model import load_model
from utils.logger import create_logger
from utils.perf_utils import pop_perf_stats
from utils.resouce_manager import ResourceManager
from utils.utils import seed_everything


resource_manager = ResourceManager()


class Evaluator:
    def __init__(self, cfg, args):
        seed_everything(cfg.get("seed", 42))

        self.output_file = Path(cfg["cfg_file"].replace("cfg/", "results/").replace("temp/", "").replace(".yaml", ".json.gz"))

        # todo resource manager
        self.cfg = cfg
        self.cfg_file = cfg["cfg_file"]
        self.model_cfg = cfg["model"]
        self.model = None
        self.dataset = resource_manager.load_dataset(**cfg["dataset"])
        self.gen_config = cfg["generation"]
        self.args = args

        self.logger = create_logger(args.get("logger", "wandb"), cfg, run_prefix=args.get("run_prefix", None), resume=args.get("resume", False))

    def process_sample(self, index):
        sample = self.dataset[index]
        input, label, data_metadata = sample["input"], sample.get("label", None), sample.get("metadata", {})
        model_output = self.model.generate(**input, gen_config=self.gen_config, output_history=True)

        sample_metrics, _ = self._compute_metrics([model_output.output], [label])

        perf_stats = pop_perf_stats(flatten=True)

        metrics = {
            **{f"sample/{k}": v for k, v in sample_metrics.items()},
            "input_length": model_output.input_length,
            "output_length": model_output.output_length,
            "nfe": model_output.nfe,
            "decoding_order": model_output.decoding_order,
            **(model_output.decoding_order_corrs if model_output.decoding_order_corrs is not None else {})
        }

        self.logger.log({
            **metrics,
            "progress": (index + 1) / len(self.dataset),
            **{f"perf/{k}": v for k, v in perf_stats.items()},
        }, index)

        return {
            "input": input["messages"], 
            "output": model_output.output, 
            "output_full": model_output.output_full,
            "label": label, 
            **metrics,
            **{f"data/metadata/{k}": v for k, v in data_metadata.items()},
            "history": model_output.history,
        }

    def generate(self):
        self.model = resource_manager.load_model(**self.model_cfg)
        return [self.process_sample(i) for i in tqdm(range(len(self.dataset)))]

    def load_results(self):
        try:
            with gzip.open(self.output_file, "rt") as f:
                outputs = json.load(f)
        except Exception as e:
            print(e)
            return None, None

        return outputs.get("outputs", None), outputs.get("metrics", None)

    def _compute_metrics(self, predictions, references, output_per_sample=False, **kwargs):
        if not (self.args.get("skip_metrics") or self.cfg.get("skip_metrics", False)):
            if not output_per_sample:
                metrics, metrics_per_sample = self.dataset.compute_metrics(predictions, references, **kwargs), []
            else:
                metrics, metrics_per_sample = self.dataset.compute_metrics(predictions, references, output_per_sample=output_per_sample, **kwargs)
        else:
            metrics, metrics_per_sample = {}, []
        
        return metrics, metrics_per_sample

    def compute_metrics(self, outputs, **kwargs):
        predictions = [output["output"] for output in outputs]
        references = [output["label"] for output in outputs if output["label"] is not None]

        output_per_sample = not any("sample/score" in k for k in outputs[0].keys())
        metrics, metrics_per_sample = self._compute_metrics(predictions, references, output_per_sample=output_per_sample, **kwargs)

        if output_per_sample:
            for o, m in zip(outputs, metrics_per_sample):
                o.update({f"sample/{k}": v for k, v in m.items()})

        return outputs, metrics

    def save_results(self, outputs, metrics):
        df = pd.DataFrame(outputs)
        df.drop(columns=["history"], inplace=True, errors="ignore")  # to save space
        self.logger.log_table("results", df)

        self.output_file.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(self.output_file, "wt") as f:
            json.dump({
                "cfg_file": self.cfg_file,
                "metrics": metrics,
                "outputs": outputs
            }, f, indent=4)

        self.logger.log({
            **metrics,
            "nfe_mean": sum(output.get("nfe", 0) for output in outputs) / len(outputs),
        })

        self.logger.finish()

        print(metrics)

    def _update_decoding_order(self, outputs):
        from model.model_utils import compute_decoding_order_correlation_from_history
        from model.llada_model import LLADA_MASK_TOKEN_ID

        self.tokenizer = resource_manager.load_tokenizer(pretrained_model_name_or_path=self.model_cfg["model_name"])
        if getattr(self.tokenizer, "mask_token_id", None) is None:
            self.tokenizer.mask_token_id = LLADA_MASK_TOKEN_ID

        for output in outputs:
            if output.get("decoding_order", None) is None or True:
                decoding_order, decoding_order_corrs = compute_decoding_order_correlation_from_history(self.tokenizer, output.get("history", []))
                output.update({
                    "decoding_order": decoding_order,
                    **decoding_order_corrs
                })
            else:
                return None
        
        return outputs


def run_main(cfg, args):
    if args.get("update_decoding_order", False):
        args["logger"] = "none"

    evaluator = Evaluator(cfg, args)

    if args.get("update_decoding_order", False):
        results = evaluator.load_results()

        if results is None:
            return

        outputs = evaluator._update_decoding_order(results["outputs"]) # temp

        if outputs is None:
            return

        metrics = results["metrics"]
        print(evaluator.output_file)
    elif args.get("compute_metrics", False):
        cfg["skip_metrics"] = False
        outputs, metrics = evaluator.load_results()

        if metrics is not None and len(metrics) > 0 and not args.get("overwrite", False):
            print(f"Metrics already computed for {evaluator.output_file}, skipping...")
            print(metrics)
            return

        if outputs is None:
            return

        outputs, metrics = evaluator.compute_metrics(outputs, allow_unsafe_eval=True)
    else:
        outputs = evaluator.generate()
        outputs, metrics = evaluator.compute_metrics(outputs)

    evaluator.save_results(outputs, metrics)


@torch.no_grad()
def main(cfg_files, args):
    if not args.get("no_progress", False):
        cfg_files = tqdm(cfg_files)

    for cfg_file in cfg_files:
        with open(cfg_file, "r") as f:
            cfg = yaml.safe_load(f)
 
        if cfg is None:
            print(f"Error loading cfg file {cfg_file}")
            continue

        cfg["cfg_file"] = cfg_file
        cfg["device"] = os.environ.get("CUDA_VISIBLE_DEVICES")
 
        run_main(cfg, args)
 
 
def parse_cfg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--cfg", type=str, required=True, nargs="+")
    parser.add_argument("--logger", type=str, default="wandb")
    parser.add_argument("--compute_metrics", action="store_true")
    parser.add_argument("--update_decoding_order", action="store_true")
    parser.add_argument("--skip_metrics", action="store_true")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument("--no_progress", action="store_true")
    args = parser.parse_args()
 
    return args.cfg, vars(args)
 
 
if __name__ == "__main__":
    main(*parse_cfg())