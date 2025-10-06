

import hashlib
import itertools
from pathlib import Path

import pandas as pd
import yaml



ALPHABET_CHARS = [chr(i) for i in range(ord("A"), ord("Z") + 1)]


class RandomMathOp:
    all_ops = ["+", "-", "*"]

    def __init__(self, rng, op, target_digits, a=None, b=None):
        self.rng = rng
        self.op = op
        self.target_digits = target_digits

        assert not (a is not None and b is not None), "Cannot specify both a and b"
        
        if b is not None:
            swap = True
            a = b
        else:
            swap = False

        if self.op == "+":
            self.a = self._generate_random_number_with_digits(self.target_digits - 1) if a is None else a
            self.b = self._generate_random_number_with_digits(self.target_digits - 1)
        elif self.op == "-":
            self.a = self._generate_random_number_with_digits(self.target_digits - 1) if a is None else a
            self.b = self._generate_random_number_with_digits(self.target_digits - 1)
        elif self.op == "*":
            self.a = self._generate_random_number_with_digits(self.target_digits // 2) if a is None else a
            self.b = self._generate_random_number_with_digits(max(self.target_digits - self.get_digit_count(self.a), 1))
        else:
            raise ValueError(f"Unsupported operation: {self.op}")
        
        if swap:
            self.a, self.b = self.b, self.a

        a = self.a if not isinstance(self.a, RandomMathOp) else self.a.target
        b = self.b if not isinstance(self.b, RandomMathOp) else self.b.target

        match self.op:
            case "+":
                self.target = a + b
            case "-":
                self.target = a - b
            case "*":
                self.target = a * b
            case _:
                raise ValueError(f"Unsupported operation: {self.op}")

    def get_digit_count(self, n):
        if isinstance(n, RandomMathOp):
            n = n.target

        return len(str(abs(n)))

    def _generate_random_number_with_digits(self, num_digits, min_value=0, max_value=9):
        return int(''.join(str(self.rng.randint(min_value if i != 0 else 1, max_value)) for i in range(num_digits)))

    def __repr__(self):
        return f"({self.a} {self.op} {self.b})"

    def get_prompt(self):
        return f"{str(self)[1:-1]} = ?"
    
    def get_target(self):
        return self.target

    def check_result(self, result):
        return str(self.target) in result.replace(",", "").replace(" ", "")

    @staticmethod
    def create_chain(rng, target_digits, num_ops, ops=None):
        if ops is None:
            ops = RandomMathOp.all_ops

        op = rng.choice(ops)

        if num_ops == 2:
            return RandomMathOp(rng, op, target_digits)

        a = RandomMathOp.create_chain(rng, target_digits, num_ops - 1, ops=ops)
        return RandomMathOp(rng, op, target_digits, a=a)
    


def _shuffle(rng, x):
    y = x[:]
    while y == x:
        rng.shuffle(y)
    return y


def _generate_domino_sequence(rng, length, start):
    sequence = [start]
    for _ in range(length - 1):  
        prev = sequence[-1]
        last_digit = prev % 10
        next_number = last_digit * 10 + rng.randint(1, 9)
        sequence.append(next_number)
    return sequence




def parse_global_config(global_config, task):
    global_config = {**global_config}
    for k, v in global_config.items():
        if isinstance(v, dict):
            v = v.get(task.split("/")[-1], v["default"])
        global_config[k] = v
    return global_config


def load_task_configs(task_config_file):
    task_config_file = Path(task_config_file)
    
    if "." not in str(task_config_file):
        task_config_file = Path(__file__).parent / "task_configs" / f"{task_config_file}.yaml"

    with open(task_config_file, "r") as f:
        task_config = yaml.safe_load(f)

    global_config = task_config.get("global_config", {})
    tasks_file = task_config.pop("tasks", None)

    with open(Path(task_config_file).parent / tasks_file, "r") as f:
        tasks = yaml.safe_load(f)

    tasks = {f"{task_config_file.stem}/{k}": v for k, v in tasks.items()}
    tasks = {task_name: {**cfg, **parse_global_config(global_config, task_name), "name": task_name} for task_name, cfg in tasks.items()}

    return tasks


def _get_task_file(split, task_name):
    return Path(__file__).parent / "output" / split / f"{task_name}.jsonl"


def str_to_seed(seed_str, offset=0):
    seed = int(hashlib.sha256(seed_str.encode()).hexdigest(), 16) + offset
    return seed % (2**16 - 1)  # Ensure seed is within the range of a 32-bit unsigned integer


def list_difference(list1, list2):
    list2 = set(list2)
    return [item for item in list1 if item not in list2]


def list_to_str(lst):
    tokens = ["[\""]

    for word in lst:
        tokens.append(str(word))
        tokens.append("\", \"")

    tokens.pop()  # Remove the last comma
    tokens.append("\"]")
    return "".join(tokens)


def sentence_to_words(sentence):
    return sentence.replace(".", "").replace(",", "").split()


def repeat_list(lst, count, repeat_type="each"):
    if repeat_type == "each":
        return list(itertools.chain.from_iterable(zip(*[lst] * count)))
    elif repeat_type == "list":
        return lst * count
    else:
        raise ValueError(f"Unknown repeat type: {repeat_type}")


def load_words_from_file(file_path):
    if isinstance(file_path, list):
        return file_path

    if file_path == "<ALPHABET_CHARS>":
        return ALPHABET_CHARS

    if file_path.endswith(".txt"):
        with open(Path(__file__).parent / file_path, "r") as f:
            words = [line.strip() for line in f if line.strip()]
    elif file_path.endswith(".yaml") or file_path.endswith(".yml"):
        with open(Path(__file__).parent / file_path, "r") as f:
            words = yaml.safe_load(f)

        if isinstance(words[0], list):
            words = [" ".join(w) for w in itertools.product(*words)]

    return words


def generate_word_lists(rng, words, num_samples, min_length=None, max_length=None, lengths=None, with_replacement=False, **kwargs):
    for _ in range(num_samples):
        if lengths:
            length = rng.choice(lengths)
        else:
            length = rng.randint(min_length, max_length)

        if with_replacement:
            x = rng.choices(words, k=length)
        else:
            x = rng.sample(words, length)
        yield x


def generate_latin_square(rng, symbols):
    size = len(symbols)
    first_row = symbols[:]
    square = [first_row]

    # Create the remaining rows by cyclically shifting the first row
    for i in range(1, size):
        next_row = first_row[i:] + first_row[:i]
        square.append(next_row)

    # Randomly shuffle rows and columns to create a more varied square
    rng.shuffle(square) # Shuffle rows
    
    # Transpose and shuffle to shuffle columns
    transposed_square = list(zip(*square))
    rng.shuffle(transposed_square)
    
    # Transpose back to original orientation
    final_square = [list(row) for row in zip(*transposed_square)]

    return final_square


def latin_square_to_str(square):
    return "\n".join(",".join(str(cell) for cell in row) for row in square)


def str_to_latin_square(text, symbols):
    size = len(symbols)
    rows = text.strip().split("\n")

    if not len(rows) == size:
        return None

    square = []
    for row in rows:
        row = [cell.strip() for cell in row.split(",")]
        if not len(row) == size:
            return None
        if set(symbols) != set(row):
            return None
        square.append(row)

    return square
