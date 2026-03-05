"""Task generator functions for ParallelBench benchmark tasks.

Each generator is registered via @register_task_generator and yields sample dicts.
"""

import numpy as np

from parallelbench.dataset.task_utils import (
    ALPHABET_CHARS,
    RandomMathOp,
    _generate_domino_sequence,
    _shuffle,
    generate_latin_square,
    generate_word_lists,
    latin_square_to_str,
    list_difference,
    list_to_str,
    repeat_list,
)
from parallelbench.utils.grammar_check import grammar_check


TASK_GENERATORS: dict[str, callable] = {}


def register_task_generator(task_type: str):
    """Decorator to register a task generator function for a given task type."""

    def decorator(func):
        TASK_GENERATORS[task_type] = func
        return func

    return decorator


def _create_mask_template(lst):
    """Create a [MASK] token template matching the length of the given list."""
    from parallelbench.dataset.task import PARALLEL_BENCH_MASK_TOKEN

    return list_to_str([PARALLEL_BENCH_MASK_TOKEN] * len(lst))


@register_task_generator("sort")
def generate_sort_task(rng, task_config):
    return (
        {
            "input": {"context": list_to_str(selected_words)},
            "answer": list_to_str(sorted(selected_words)),
            "metadata": {
                "length": len(selected_words),
            },
        }
        for selected_words in generate_word_lists(rng, **task_config)
    )


@register_task_generator("shuffle")
def generate_shuffle_task(rng, task_config):
    return (
        {
            "input": {"context": list_to_str(selected_words)},
            "answer": {
                "input": selected_words,
                "example": list_to_str(_shuffle(rng, selected_words)),
            },
            "output_format": _create_mask_template(selected_words),
            "metadata": {
                "length": len(selected_words),
            },
        }
        for selected_words in generate_word_lists(rng, **task_config)
    )


@register_task_generator("copy")
def generate_copy_task(rng, task_config):
    return (
        {
            "input": {"context": list_to_str(selected_words)},
            "answer": list_to_str(selected_words),
            "output_format": _create_mask_template(selected_words),
            "metadata": {
                "length": len(selected_words),
            },
        }
        for selected_words in generate_word_lists(rng, **task_config)
    )


@register_task_generator("reverse")
def generate_reverse_task(rng, task_config):
    return (
        {
            "input": {"context": list_to_str(selected_words)},
            "answer": list_to_str(list(reversed(selected_words))),
            "metadata": {
                "length": len(selected_words),
            },
        }
        for selected_words in generate_word_lists(rng, **task_config)
    )


@register_task_generator("repeat")
def generate_repeat_task(rng, task_config):
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
            "metadata": {"length": len(lst), "count": count},
        }


@register_task_generator("insert")
def generate_insert_task(rng, task_config):
    for input_list in generate_word_lists(rng, **task_config):
        word_to_insert = rng.choice(list_difference(task_config["words"], input_list))

        input = {"context": list_to_str(input_list), "word": word_to_insert}

        index_to_insert = rng.randint(0, len(input_list))

        target_list = input_list[:]
        target_list.insert(index_to_insert, word_to_insert)

        assert len(set(target_list)) == len(target_list), (
            "Target list must not contain duplicates"
        )

        if not task_config["random_index"]:
            input["index"] = index_to_insert
            answer = list_to_str(target_list)
        else:
            index_to_insert = None
            answer = {
                "input": input_list,
                "word": word_to_insert,
                "example": list_to_str(target_list),
            }

            assert len(set(input_list)) == len(input_list), (
                "Target list must not contain duplicates"
            )
            assert word_to_insert not in input_list, (
                "Inserted word must not be in the input list"
            )

        yield {
            "input": input,
            "answer": answer,
            "output_format": _create_mask_template(target_list),
            "metadata": {
                "length": len(input_list),
                "index": index_to_insert,
                "word": word_to_insert,
            },
        }


@register_task_generator("remove")
def generate_remove_task(rng, task_config):
    for input_list in generate_word_lists(rng, **task_config):
        input = {
            "context": list_to_str(input_list),
        }

        index_to_remove = rng.randint(0, len(input_list) - 1)

        target_list = input_list[:]
        target_list.pop(index_to_remove)

        assert len(set(target_list)) == len(target_list), (
            "Target list must not contain duplicates"
        )

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
            },
        }


@register_task_generator("replace")
def generate_replace_task(rng, task_config):
    for input_list in generate_word_lists(rng, **task_config):
        new_word = rng.choice(list_difference(task_config["words"], input_list))

        input = {"context": list_to_str(input_list), "word": new_word}

        index_to_replace = rng.randint(0, len(input_list) - 1)

        target_list = input_list[:]
        target_list[index_to_replace] = new_word

        assert len(set(target_list)) == len(target_list), (
            "Target list must not contain duplicates"
        )

        if not task_config["random_index"]:
            input["index"] = index_to_replace
            answer = list_to_str(target_list)
        else:
            index_to_replace = None
            answer = {
                "input": input_list,
                "word": new_word,
                "example": list_to_str(target_list),
            }

            assert len(set(input_list)) == len(input_list), (
                "Target list must not contain duplicates"
            )
            assert new_word not in input_list, (
                "Inserted word must not be in the input list"
            )

        yield {
            "input": input,
            "answer": answer,
            "output_format": _create_mask_template(target_list),
            "metadata": {
                "length": len(input_list),
                "index": index_to_replace,
                "word": new_word,
            },
        }


@register_task_generator("domino")
def generate_domino_task(rng, task_config):
    min_length = task_config["min_length"]
    max_length = task_config["max_length"]
    num_samples = task_config["num_samples"]

    for _ in range(num_samples):
        length = rng.randint(min_length, max_length)
        start = rng.randint(1, 9) * 10 + rng.randint(1, 9)

        input = {"length": length, "start": start}

        answer = {
            **input,
            "example": list_to_str(_generate_domino_sequence(rng, length, start)),
        }

        yield {
            "input": input,
            "answer": answer,
            "metadata": {
                "length": length,
            },
        }


@register_task_generator("math_op")
def generate_math_op_task(rng, task_config):
    lengths = task_config["lengths"]
    num_samples = task_config["num_samples"]
    num_ops = task_config["num_ops"]

    for _ in range(num_samples):
        length = rng.choice(lengths)

        op = RandomMathOp.create_chain(
            rng, target_digits=length, num_ops=num_ops, ops=task_config.get("ops")
        )

        yield {
            "input": {"equation": op.get_prompt()},
            "answer": {"result": str(op.get_target())},
            "metadata": {"length": length, "true_length": len(str(op.get_target()))},
        }


@register_task_generator("latin_square")
def generate_latin_square_task(rng, task_config):
    size = task_config["size"]
    num_samples = task_config["num_samples"]

    all_symbols = ALPHABET_CHARS + [str(i) for i in (range(0, 10))]

    for _ in range(num_samples):
        symbols = rng.sample(all_symbols, size)

        latin_square = generate_latin_square(rng, symbols)

        yield {
            "input": {"size": size, "symbols": list_to_str(symbols).replace('"', "")},
            "answer": {
                "symbols": symbols,
                "example": latin_square_to_str(latin_square),
            },
            "metadata": {
                "length": size,
            },
        }


@register_task_generator("rec_cumsum")
def generate_rec_cumsum_task(rng, task_config):
    return (
        {
            "input": {"list": list_to_str(numbers).replace('"', "")},
            "answer": list_to_str(np.cumsum(numbers)).replace('"', ""),
            "metadata": {
                "length": len(numbers),
            },
        }
        for numbers in generate_word_lists(
            rng,
            list(range(1, 10)),
            num_samples=task_config["num_samples"],
            lengths=task_config["lengths"],
            with_replacement=True,
        )
    )


@register_task_generator("summary")
def generate_summary_task(rng, task_config):
    source = task_config["source"]
    num_samples = task_config["num_samples"]

    if source == "samsum":
        from datasets import load_dataset

        dataset = load_dataset("knkarthick/samsum", split="test")
        dataset = dataset.shuffle(seed=rng.randint(0, int(1e9)))

        i = 0
        for sample in dataset:
            if grammar_check(sample["dialogue"]):
                yield {
                    "input": {"text": sample["dialogue"]},
                    "answer": {
                        "text": sample["dialogue"],
                        "summary": sample["summary"],
                    },
                    "metadata": {
                        "length": 1,
                    },
                }
                i += 1
                if i >= num_samples:
                    break
            else:
                print("Skipping non-grammatical sample")
    else:
        raise ValueError(f"Unknown source: {source}")


@register_task_generator("paraphrase")
def generate_paraphrase_task(rng, task_config):
    source = task_config["source"]
    num_samples = task_config["num_samples"]

    if source == "chatgpt-paraphrases":
        from datasets import load_dataset

        dataset = load_dataset("humarin/chatgpt-paraphrases", split="train")
        dataset = dataset.shuffle(seed=rng.randint(0, int(1e9)))

        i = 0
        for sample in dataset:
            if grammar_check(sample["text"]):
                yield {
                    "input": {"text": sample["text"]},
                    "answer": {
                        "text": sample["text"],
                        "examples": sample["paraphrases"],
                    },
                    "metadata": {
                        "length": 1,
                    },
                }
                i += 1
                if i >= num_samples:
                    break
            else:
                print("Skipping non-grammatical sample")
    else:
        raise ValueError(f"Unknown source: {source}")
