
from pathlib import Path
import argparse
import hashlib
import itertools
import random
import numpy as np
import pandas as pd

from tqdm import tqdm
import yaml


from datasets import Dataset, concatenate_datasets


from dataset.parallel_bench.data.task_utils import ALPHABET_CHARS, RandomMathOp, _generate_domino_sequence, _get_task_file, _shuffle, generate_latin_square, generate_word_lists, latin_square_to_str, list_difference, list_to_str, load_task_configs, load_words_from_file, repeat_list, str_to_seed
from utils.grammar_check import grammar_check


DEFAULT_SEED = 42


PARALLEL_BENCH_MASK_TOKEN = "[MASK]"


def _get_list_output_format(lst):
    return list_to_str([PARALLEL_BENCH_MASK_TOKEN] * len(lst))


def load_task(split, task_name):
    task_file = _get_task_file(split, task_name)
    task_config_file = task_file.parent / "task_config.yaml"

    with open(task_config_file, "r") as f:
        task_config = yaml.safe_load(f)

    task_config = task_config[task_name]
    task = Dataset.from_pandas(pd.read_json(path_or_buf=task_file, lines=True))

    return task, task_config


def generate_parallel_bench_sort_task(rng, task_config):
    return ({
        "input": {"context": list_to_str(selected_words)},
        "answer": list_to_str(sorted(selected_words)),
        "metadata": {
            "length": len(selected_words),
        }
    } for selected_words in generate_word_lists(rng, **task_config))


def generate_parallel_bench_shuffle_task(rng, task_config):
    return ({
        "input": {"context": list_to_str(selected_words)},
        "answer": {"input": selected_words, "example": list_to_str(_shuffle(rng, selected_words))},
        "output_format": _get_list_output_format(selected_words),
        "metadata": {
            "length": len(selected_words),
        }
    } for selected_words in generate_word_lists(rng, **task_config))


def generate_parallel_bench_copy_task(rng, task_config):
    return ({
        "input": {"context": list_to_str(selected_words)},
        "answer": list_to_str(selected_words),
        "output_format": _get_list_output_format(selected_words),
        "metadata": {
            "length": len(selected_words),
        }
    } for selected_words in generate_word_lists(rng, **task_config))


def generate_parallel_bench_reverse_task(rng, task_config):
    return ({
        "input": {"context": list_to_str(selected_words)},
        "answer": list_to_str(list(reversed(selected_words))),
        "metadata": {
            "length": len(selected_words),
        }
    } for selected_words in generate_word_lists(rng, **task_config))


def generate_parallel_bench_repeat_task(rng, task_config):
    repeat_type = task_config["repeat_type"]
    num_samples = task_config["num_samples"]
    min_length = task_config["min_length"]
    max_length = task_config["max_length"]
    repeat_counts = task_config["repeat_counts"]
    words = task_config["words"]

    for _ in range(num_samples):
        count = rng.choice(repeat_counts)
        length = rng.randint(min_length, max_length // count)
        lst = rng.sample(words, length)

        yield {
            "input": {"context": list_to_str(lst), "count": count},
            "answer": list_to_str(repeat_list(lst, count, repeat_type)),
            "metadata": {
                "length": len(lst),
                "count": count
            }
        }

def generate_parallel_bench_insert_task(rng, task_config):
    # max_length = task_config.pop("max_length") - 1  # to leave space for the inserted word
    # min_length = task_config.pop("min_length") - 1  # to leave space for the inserted word
    
    for input_list in generate_word_lists(rng, **task_config):  # , min_length=min_length, max_length=max_length)
        word_to_insert = rng.choice(list_difference(task_config["words"], input_list))

        input = {
            "context": list_to_str(input_list),
            "word": word_to_insert
        }

        index_to_insert = rng.randint(0, len(input_list))

        target_list = input_list[:]
        target_list.insert(index_to_insert, word_to_insert)

        assert len(set(target_list)) == len(target_list), "Target list must not contain duplicates"

        if not task_config["random_index"]:
            input["index"] = index_to_insert
            answer = list_to_str(target_list)
        else:
            index_to_insert = None
            answer = {"input": input_list, "word": word_to_insert, "example": list_to_str(target_list)}

            assert len(set(input_list)) == len(input_list), "Target list must not contain duplicates"
            assert word_to_insert not in input_list, "Inserted word must not be in the input list"

        yield {
            "input": input,
            "answer": answer,
            "output_format": _get_list_output_format(target_list),
            "metadata": {
                "length": len(input_list),
                "index": index_to_insert,
                "word": word_to_insert
            }
        }


def generate_parallel_bench_remove_task(rng, task_config):
    for input_list in generate_word_lists(rng, **task_config):
        input = {
            "context": list_to_str(input_list),
        }

        index_to_remove = rng.randint(0, len(input_list) - 1)

        target_list = input_list[:]
        target_list.pop(index_to_remove)

        assert len(set(target_list)) == len(target_list), "Target list must not contain duplicates"

        if not task_config["random_index"]:
            input["index"] = index_to_remove
            answer = list_to_str(target_list)
        else:
            index_to_remove = None
            answer = {"input": input_list, "example": list_to_str(target_list)}

        yield {
            "input": input,
            "answer": answer,
            "metadata": {
                "length": len(input_list),
                "index": index_to_remove,
            }
        }


def generate_parallel_bench_replace_task(rng, task_config):
    for input_list in generate_word_lists(rng, **task_config):
        new_word = rng.choice(list_difference(task_config["words"], input_list))

        input = {
            "context": list_to_str(input_list),
            "word": new_word
        }

        index_to_replace = rng.randint(0, len(input_list) - 1)

        target_list = input_list[:]
        target_list[index_to_replace] = new_word

        assert len(set(target_list)) == len(target_list), "Target list must not contain duplicates"

        if not task_config["random_index"]:
            input["index"] = index_to_replace
            answer = list_to_str(target_list)
        else:
            index_to_replace = None
            answer = {"input": input_list, "word": new_word, "example": list_to_str(target_list)}

            assert len(set(input_list)) == len(input_list), "Target list must not contain duplicates"
            assert new_word not in input_list, "Inserted word must not be in the input list"

        yield {
            "input": input,
            "answer": answer,
            "output_format": _get_list_output_format(target_list),
            "metadata": {
                "length": len(input_list),
                "index": index_to_replace,
                "word": new_word,
            }
        }


def generate_parallel_bench_domino_task(rng, task_config):
    min_length = task_config["min_length"]
    max_length = task_config["max_length"]
    num_samples = task_config["num_samples"]

    for _ in range(num_samples):
        length = rng.randint(min_length, max_length)
        start = rng.randint(1, 9) * 10 + rng.randint(1, 9)

        input = {
            "length": length,
            "start": start
        }

        answer = {
            **input,
            "example": list_to_str(_generate_domino_sequence(rng, length, start)),
        }

        yield {
            "input": input,
            "answer": answer,
            "metadata": {
                "length": length,
            }
        }


def generate_parallel_bench_math_op_task(rng, task_config):
    lengths = task_config["lengths"]
    num_samples = task_config["num_samples"]
    num_ops = task_config["num_ops"]

    for _ in range(num_samples):
        length = rng.choice(lengths)

        op = RandomMathOp.create_chain(rng, target_digits=length, num_ops=num_ops, ops=task_config.get("ops"))

        yield {
            "input": {"equation": op.get_prompt()},
            "answer": {"result": str(op.get_target())},
            "metadata": {
                "length": length,
                "true_length": len(str(op.get_target()))
            }
        }


def generate_latin_square_task(rng, task_config):
    size = task_config["size"]
    num_samples = task_config["num_samples"]

    all_symbols = ALPHABET_CHARS + [str(i) for i in (range(0, 10))]

    for _ in range(num_samples):
        symbols = rng.sample(all_symbols, size)

        latin_square = generate_latin_square(rng, symbols)

        yield {
            "input": {"size": size, "symbols": list_to_str(symbols).replace('"', "")},
            "answer": {"symbols": symbols, "example": latin_square_to_str(latin_square)},
            "metadata": {
                "length": size,
            }
        }


def generate_rec_cumsum_task(rng, task_config):
    return ({
        "input": {"list": list_to_str(numbers).replace('"', "")},
        "answer": list_to_str(np.cumsum(numbers)).replace('"', ""),
        "metadata": {
            "length": len(numbers),
        }
    } for numbers in generate_word_lists(rng, list(range(1, 10)), num_samples=task_config["num_samples"], lengths=task_config["lengths"], with_replacement=True))


def generate_summary_task(rng, task_config):
    source = task_config["source"]
    num_samples = task_config["num_samples"]

    if source == "samsum":
        from datasets import load_dataset
        # dataset = concatenate_datasets([load_dataset("samsum", split="validation"), load_dataset("samsum", split="test")])
        dataset = load_dataset("knkarthick/samsum", split="test")

        dataset = dataset.shuffle(seed=rng.randint(0, 1e9))  # .select(range(min(num_samples, len(dataset))))

        i = 0
        for sample in dataset:
            if grammar_check(sample["dialogue"]):
                yield {
                    "input": {"text": sample["dialogue"]},
                    "answer": {"text": sample["dialogue"], "summary": sample["summary"]},
                    "metadata": {
                        "length": 1,
                    }
                }
                i += 1
                if i >= num_samples:
                    break
            else:
                print("Skipping non-grammatical sample")
    else:
        raise ValueError(f"Unknown source: {source}")
    

def generate_paraphrase_task(rng, task_config):
    source = task_config["source"]
    num_samples = task_config["num_samples"]

    if source == "chatgpt-paraphrases":
        from datasets import load_dataset
        dataset = load_dataset("humarin/chatgpt-paraphrases", split="train")
        dataset = dataset.shuffle(seed=rng.randint(0, 1e9))  # .select(range(min(num_samples, len(dataset))))

        i = 0
        for sample in dataset:
            if grammar_check(sample["text"]):
                yield {
                    "input": {"text": sample["text"]},
                    "answer": {"text": sample["text"], "examples": sample["paraphrases"]},
                    "metadata": {
                        "length": 1,
                    }
                }
                i += 1
                if i >= num_samples:
                    break
            else:
                print("Skipping non-grammatical sample")
    else:
        raise ValueError(f"Unknown source: {source}")


def generate_parallel_bench_task_random(rng, task_config, infinite=False):
    task_config = {**task_config}

    if infinite:
        task_config["num_samples"] = int(1e10)

    if task_config["type"] == "sort":
        yield from generate_parallel_bench_sort_task(rng, task_config)
    elif task_config["type"] == "shuffle":
        yield from generate_parallel_bench_shuffle_task(rng, task_config)
    elif task_config["type"] == "copy":
        yield from generate_parallel_bench_copy_task(rng, task_config)
    elif task_config["type"] == "reverse":
        yield from generate_parallel_bench_reverse_task(rng, task_config)
    elif task_config["type"] in "repeat":
        yield from generate_parallel_bench_repeat_task(rng, task_config)
    elif task_config["type"] in "insert":
        yield from generate_parallel_bench_insert_task(rng, task_config)
    elif task_config["type"] in "remove":
        yield from generate_parallel_bench_remove_task(rng, task_config)
    elif task_config["type"] in "replace":
        yield from generate_parallel_bench_replace_task(rng, task_config)
    elif task_config["type"] in "domino":
        yield from generate_parallel_bench_domino_task(rng, task_config)
    elif task_config["type"] in "math_op":
        yield from generate_parallel_bench_math_op_task(rng, task_config)
    elif task_config["type"] == "latin_square":
        yield from generate_latin_square_task(rng, task_config)
    elif task_config["type"] == "rec_cumsum":
        yield from generate_rec_cumsum_task(rng, task_config)
    elif task_config["type"] == "summary":
        yield from generate_summary_task(rng, task_config)
    elif task_config["type"] == "paraphrase":
        yield from generate_paraphrase_task(rng, task_config)
    else:
        raise ValueError(f"Unknown task type: {task_config['type']}")


def create_parallel_bench_task_random(rng, task):
    return list(generate_parallel_bench_task_random(rng, task))


def create_parallel_bench_task_random_samples_per_length(rng, task):
    samples_per_length = task["samples_per_length"]
    num_samples = task["num_samples"]
    assert num_samples % samples_per_length == 0, "num_samples must be divisible by samples_per_length"
    num_buckets = num_samples // samples_per_length

    data_per_length = {}

    finished = False
    for sample in tqdm(generate_parallel_bench_task_random(rng, task, infinite=True)):
        length = sample["metadata"]["length"] if task["type"] != "repeat" else sample["metadata"]["count"]
        if length not in data_per_length:
            data_per_length[length] = []

        if len(data_per_length[length]) < samples_per_length:
            data_per_length[length].append(sample)

        if sum(len(data) == samples_per_length for data in data_per_length.values()) == num_buckets:
            data_per_length = {k: v for k, v in data_per_length.items() if len(v) == samples_per_length}  # exclude too short length buckets
            finished = True
            break

    assert finished
    lengths = sorted(data_per_length.keys())
    data = sum([data_per_length[l] for l in lengths], [])
    assert len(data) == num_samples, f"Expected {num_samples} samples, got {len(data)}"

    return data


def _create_task(rng, task):
    print(f"Creating task {task['name']} with seed {task['seed']}...")
    if task.get("samples_per_length", 0) > 0:
        data = create_parallel_bench_task_random_samples_per_length(rng, task)
    else:
        data = create_parallel_bench_task_random(rng, task)
    return data


def create_parallel_bench_task(split, task, output_file, rng=None, no_save=False):
    if not output_file:
        output_file = _get_task_file(split, task_name=task["name"])

    if "seed" not in task:
        task["seed"] = str_to_seed(task["name"].split("/")[-1], DEFAULT_SEED)
    else:
        task["seed"] = str_to_seed(task["name"].split("/")[-1], task["seed"])

    if rng is None:
        rng = random.Random(task["seed"])

    if "words" in task:
        words_file = task["words"]
        task["words"] = load_words_from_file(task["words"])
    else:
        words_file = None

    data = _create_task(rng, task)

    if task.get("icl_example_count", 0) > 0:
        icl_datasets = [create_parallel_bench_task_random(rng=rng, task={**task, "icl_example_count": 0}) for t in range(task["icl_example_count"])]

        for i, sample in enumerate(data):
            sample["input"]["icl_examples"] = [icl_dataset[i] for icl_dataset in icl_datasets]

    if not no_save:
        if not isinstance(data, pd.DataFrame):
            data = pd.DataFrame(data)

        Path(output_file).parent.mkdir(parents=True, exist_ok=True)
        data.to_json(output_file, orient="records", lines=True)

        out_task_config_file = Path(output_file).parent / "task_config.yaml"

        if out_task_config_file.exists():
            with open(out_task_config_file, "r") as f:
                out_task_config = yaml.safe_load(f)
        else:
            out_task_config = {}
        
        out_task_config[task["name"]] = task

        if words_file is not None:
            task["words"] = words_file

        with open(out_task_config_file, "w") as f:
            yaml.dump(out_task_config, f)
    else:
        return data


def main(task, **kwargs):
    # if task == ["all"]:
    #     with open(Path(__file__).parent / "task_config.yaml", "r") as f:
    #         task_config = yaml.safe_load(f)

    #     task = list(task_config.keys())

    loaded_tasks = []

    for t in task:
        if t.endswith("/all"):
            t = t[:-len("/all")]
            split = t.split("/")[0]
            tasks = list(load_task_configs(t).values())
            loaded_tasks.extend(list(zip([split] * len(tasks), tasks)))
        else:
            assert False

    for split, t in loaded_tasks:
        create_parallel_bench_task(split=split, task=t, **kwargs)


def parse_args():
    parser = argparse.ArgumentParser(description="Create a DLLM task.")
    parser.add_argument("--task", type=str, nargs="+", required=True, help="Name of the task to create.")
    parser.add_argument("--output_file", type=str, required=False, help="Output file to save the task data.")
    args = parser.parse_args()
    return vars(args)


if __name__ == "__main__":
    main(**parse_args())
