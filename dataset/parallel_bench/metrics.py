
import re
import json

import evaluate
from dataset.parallel_bench.data.task_utils import str_to_latin_square
from utils.grammar_check import grammar_check


def _parse_list(input_str, strict=False):
    """Parse a string representation of a list into an actual list."""

    if "[" in input_str:
        input_str = input_str.rsplit("[", 1)[1]
    elif strict:
        return None

    if "]" in input_str:
        input_str = input_str.split("]", 1)[0]
    elif strict:
        return None

    if not strict:
        return [item.strip("\" ") for item in input_str.strip("[]").split(",") if item.strip()]
    else:
        items = input_str.split(", ")
        for i in range(len(items)):
            item = items[i]

            if item.startswith("\"") and item.endswith("\""):
                item = item[1:-1]
            else:
                return None
            
            if "\"" in item:
                return None
            
            items[i] = item

        return items
    

class Metric:
    pass


def sentence_to_words(sentence):
    word_map = {
        "an": "a",
        "An": "A",
    }

    words = sentence.replace(".", "").replace(",", "").split()
    words = [word_map.get(word, word) for word in words]
    return words


def _extract_numbers_from_str(input_str):
    # regex to find all numbers
    import re
    numbers = re.findall(r"\d+", input_str)
    return [int(num) for num in numbers]


def list_match_score(prediction, ground_truth, strict=False):
    prediction_list = _parse_list(prediction, strict)
    gt_list = _parse_list(ground_truth)

    if prediction_list is None:
        return 0.0

    return float(prediction_list == gt_list)


def math_op_score(prediction, ground_truth, strict=False):
    prediction = prediction.replace(",", "").strip()
    ground_truth = int(ground_truth.split()[-1]) if not isinstance(ground_truth, dict) else int(ground_truth["result"])

    if strict:
        try:
            prediction = int(prediction)
        except ValueError:
            return 0.0

        return prediction == ground_truth
    else:
        return f"{ground_truth:.0f}" in prediction


def list_shuffle_score(prediction, ground_truth, strict=False):
    prediction_list = _parse_list(prediction, strict)
    gt_list = ground_truth["input"][:]

    if prediction_list is None:
        return 0.0

    return float(
        (sorted(prediction_list) == sorted(gt_list)) and  # Check if both lists are equal when sorted
        (prediction_list != gt_list)  # Ensure the prediction is different from the ground truth
    )


def random_insert_score(prediction_list, ground_truth_list, word, n_words=1):
    if prediction_list is None or ground_truth_list is None:
        return 0.0
    
    prediction_list = prediction_list[:]
    ground_truth_list = ground_truth_list[:]
    
    if word is not None:
        words = [word] if isinstance(word, str) else word

        for word in words:
            try:
                prediction_list.remove(word)
            except ValueError:
                return 0.0

        return float(prediction_list == ground_truth_list)
    else:
        # remove one random word from gt is equal to insert one random word to pred
        return random_remove_score(prediction_list=ground_truth_list, ground_truth_list=prediction_list, n_words=n_words)


def list_random_insert_score(prediction, ground_truth, strict=False):
    prediction_list = _parse_list(prediction, strict=strict)
    gt_list = ground_truth["input"]
    word = ground_truth["word"]
    n_words = ground_truth.get("n_words", 1)
    return random_insert_score(prediction_list, gt_list, word, n_words=n_words)


def sentence_random_insert_score(prediction, ground_truth, strict=False):
    prediction_list = sentence_to_words(prediction)
    
    if isinstance(ground_truth, dict):
        ground_truth_list = sentence_to_words(ground_truth["input"])
        word = ground_truth.get("word")
        n_words = ground_truth.get("n_words", 1)
    else:
        ground_truth_list = sentence_to_words(ground_truth)
        n_words = 1
        word = None

    return random_insert_score(prediction_list, ground_truth_list, word, n_words=n_words)


def random_remove_score(prediction_list, ground_truth_list, n_words=1):
    if prediction_list is None or ground_truth_list is None:
        return 0.0
    
    prediction_list = prediction_list[:]
    ground_truth_list = ground_truth_list[:]

    if len(prediction_list) != len(ground_truth_list) - n_words:
        return 0.0

    deleted_words = set(ground_truth_list) - set(prediction_list)
    if len(deleted_words) != n_words:
        return 0.0

    for deleted_word in deleted_words:
        ground_truth_list.remove(deleted_word)

    return float(prediction_list == ground_truth_list)


def list_random_remove_score(prediction, ground_truth, strict=False):
    prediction_list = _parse_list(prediction, strict=strict)
    ground_truth_list = ground_truth["input"][:]
    n_words = ground_truth.get("n_words", 1)
    return random_remove_score(prediction_list, ground_truth_list, n_words=n_words)


def sentence_random_remove_score(prediction, ground_truth, strict=False):
    prediction_list = sentence_to_words(prediction)

    if isinstance(ground_truth, dict):
        ground_truth_list = sentence_to_words(ground_truth["input"])
        n_words = ground_truth.get("n_words", 1)
    else:
        ground_truth_list = sentence_to_words(ground_truth)
        n_words = 1

    return random_remove_score(prediction_list, ground_truth_list, n_words=n_words)


def random_replace_score(prediction_list, ground_truth_list, new_word=None, n_words=1):
    if prediction_list is None or ground_truth_list is None:
        return 0.0
    
    prediction_list = prediction_list[:]
    ground_truth_list = ground_truth_list[:]

    if len(prediction_list) != len(ground_truth_list):
        return 0.0

    if new_word is not None:
        try:
            replace_index = prediction_list.index(new_word)
        except ValueError:
            return 0.0
        
        ground_truth_list[replace_index] = new_word

        return float(prediction_list == ground_truth_list)
    else:
        # replace with any word
        num_diffs = sum(1 for p, g in zip(prediction_list, ground_truth_list) if p != g)
        return float(num_diffs == n_words)


def list_random_replace_score(prediction, ground_truth, strict=False):
    prediction_list = _parse_list(prediction, strict=strict)
    ground_truth_list = ground_truth["input"]
    word = ground_truth["word"]
    n_words = ground_truth.get("n_words", 1)

    return random_replace_score(prediction_list, ground_truth_list, new_word=word, n_words=n_words)


def sentence_random_replace_score(prediction, ground_truth, strict=False):
    prediction_list = sentence_to_words(prediction)
    if isinstance(ground_truth, dict):
        ground_truth_list = sentence_to_words(ground_truth["input"])
        word = ground_truth.get("word")
        n_words = ground_truth.get("n_words", 1)
    else:
        ground_truth_list = sentence_to_words(ground_truth)
        word = None
        n_words = 1

    return random_replace_score(prediction_list, ground_truth_list, new_word=word, n_words=n_words)


def sentence_replace_all_with_unique_random_score(prediction, ground_truth, strict=False):
    prediction_list = sentence_to_words(prediction)
    ground_truth_list = sentence_to_words(ground_truth["input"])
    old_word = ground_truth["word"]
    n_replace = ground_truth["n_replace"]

    metrics = {
        "score": 0.0,
        "score_loose": 0.0
    }

    if len(prediction_list) != len(ground_truth_list):
        return metrics

    num_occurences = 0
    num_unreplaced = 0
    new_words = []

    replaced_unrelated = False
    for pred_word, gt_word in zip(prediction_list, ground_truth_list):
        if pred_word == gt_word:
            if gt_word == old_word:
                # did not replaced old word
                num_occurences += 1
                num_unreplaced += 1
            else:
                # did not replace unrelated word
                pass
        else:
            if gt_word == old_word:
                # replaced old word
                num_occurences += 1
                new_words.append(pred_word)
            else:
                replaced_unrelated = True
            
    if len(new_words) != len(set(new_words)):
        # replacements are not unique
        return metrics
    
    if n_replace is None:
        metrics["score_loose"] = float(num_unreplaced == 0)
    elif n_replace < 0:
        metrics["score_loose"] = float(num_unreplaced == abs(n_replace))
    else:
        metrics["score_loose"] = float(num_occurences - num_unreplaced == n_replace)

    if not replaced_unrelated:
        metrics["score"] = metrics["score_loose"]

    return metrics


def domino_score(prediction, ground_truth, strict=False):
    if strict:
        prediction_list = _parse_list(prediction, strict=True)

        if prediction_list is None:
            return 0.0
        
        try:
            prediction_list = [int(v) for v in prediction_list]
        except ValueError:
            return 0.0
    else:
        prediction_list = _extract_numbers_from_str(prediction)

    length = ground_truth["length"]
    start = ground_truth["start"]
    
    if length != len(prediction_list):
        return 0.0

    if start != prediction_list[0]:
        return 0.0
    
    for number in prediction_list:
        if not (number > 10 and number < 100):
            return 0.0
    
    for i in range(length - 1):
        prev, next = prediction_list[i], prediction_list[i + 1]
        prev_last_digit = prev % 10
        next_first_digit = next // 10

        if prev_last_digit != next_first_digit:
            return 0.0

    return 1.0


def latin_square_score(prediction, ground_truth, strict=False):
    symbols = set(ground_truth["symbols"])
    first_row = ground_truth.get("first_row")
    size = len(symbols)

    # filter valid lines first
    pred_rows = [line.strip() for line in prediction.split("\n") if line.strip()]
    pred_rows = [line for line in pred_rows if set([cell.strip() for cell in line.split(",")]) == symbols]

    if len(pred_rows) < size:
        return 0.0
    
    pred_rows = pred_rows[:size]
    pred_rows = [line.split(",") for line in pred_rows]
    pred_rows = [[cell.strip() for cell in line] for line in pred_rows]

    rows = pred_rows
    cols = list(zip(*pred_rows))

    if any(set(row) != symbols for row in rows) or any(set(col) != symbols for col in cols):
        return 0.0
    
    if first_row is not None and rows[0] != first_row:
        return 0.0

    return 1.0


def sentence_to_words_score(prediction, ground_truth):
    words = ground_truth["words"]
    
    inclusion_score = all(word in prediction for word in words)
    grammar_score = grammar_check(prediction)
    score = inclusion_score and grammar_score

    return {
        "inclusion_score": float(inclusion_score),
        "grammar_score": float(grammar_score),
        "score": float(score)
    }


def grammar_score(prediction, ground_truth):
    return {
        "score": float(grammar_check(prediction))
    }

def startwith_score(prediction, ground_truth):
    grammar_score = float(grammar_check(prediction))
    startswith_score = float(prediction.strip().strip('"').startswith(ground_truth["startwith"]))

    return {
        "grammar_score": grammar_score,
        "startswith_score": startswith_score,
        "score": grammar_score * startswith_score
    }


def regex_match_score(prediction, ground_truth):
    pattern = ground_truth["pattern"]
    score = bool(re.fullmatch(pattern, prediction.strip()))

    return {
        "score": float(score)
    }


def text_to_regex_score(prediction, ground_truth, strict=False):
    prediction = prediction.split("```regex" if "```regex" in prediction else "```", 1)[-1].strip()
    prediction = prediction.split("```")[0].strip()

    positive_examples = ground_truth["positive_examples"]
    negative_examples = ground_truth["negative_examples"]

    try:
        prog = re.compile(prediction)
    except Exception as e:
        return 0.0

    for positive in positive_examples:
        if prog.fullmatch(positive) is None:
            return 0.0

    for negative in negative_examples:
        if prog.fullmatch(negative) is not None:
            return 0.0

    return 1.0


def json_syntax_score(prediction, ground_truth=None, strict=False):
    prediction = prediction.split("```json" if "```json" in prediction else "```", 1)[-1].strip()
    prediction = prediction.split("```")[0].strip()

    if prediction == "":
        return 0.0

    try:
        json.loads(prediction)
        return 1.0
    except Exception as e:
        return 0.0


class SummaryScore(Metric):
    def __init__(self):
        self.rouge = None

    def load(self):
        if self.rouge is None:
            self.rouge = evaluate.load('rouge')

    def __call__(self, prediction, ground_truth, strict=False):
        self.load()

        prediction = prediction.strip()
        summary = ground_truth["summary"].strip()

        rouge = self.rouge.compute(predictions=[prediction], references=[summary], use_stemmer=True)
        grammar = float(grammar_check(prediction)) if prediction != "" else 0.0

        metrics = {
            "rouge1_score": rouge["rouge1"],
            "rouge2_score": rouge["rouge2"],
            "rougeL_score": rouge["rougeL"],
            "grammar_score": grammar,
            "score": rouge["rougeL"] * grammar
        }

        metrics = {k: float(v) for k, v in metrics.items()}

        return metrics


class ParaphraseScore(Metric):
    def __init__(self):
        self.bleu = None
        self.bertscore = None

    def load(self):
        if self.bleu is None:
            self.bleu = evaluate.load('bleu')
        if self.bertscore is None:
            self.bertscore = evaluate.load('bertscore')

    def __call__(self, prediction, ground_truth, strict=False):
        self.load()

        prediction = prediction.strip()
        text = ground_truth["text"]

        if prediction != "":
            inv_bleu = 1 - self.bleu.compute(predictions=[prediction], references=[[text]])["bleu"]
            bertscore = self.bertscore.compute(predictions=[prediction], references=[text], lang="en")
            bertscore = bertscore["f1"][0]
            grammar = float(grammar_check(prediction))
        else:
            inv_bleu = 0.0
            bertscore = 0.0
            grammar = 0.0

        metrics = {
            "inv_bleu_score": inv_bleu,
            "bertscore_score": bertscore,
            "grammar_score": grammar,
            "score": inv_bleu * bertscore * grammar,
        }

        metrics = {k: float(v) for k, v in metrics.items()}

        return metrics


def parse_sudoku(sudoku_str, n=4, strict=False):
    assert n == 4

    sudoku_str = sudoku_str.strip()

    if not strict:
        sudoku_str = sudoku_str.replace(" ", "").replace(",", "")
        lines = [line.strip() for line in sudoku_str.split("\n") if line.strip()]
        grid = []
        for line in lines:
            row = [int(num) for num in re.findall(r"\d", line)]
            if len(row) == n:
                grid.append(row)

        if len(grid) != n:
            grid = None
    else:
        lines = [line.strip() for line in sudoku_str.split("\n") if line.strip()]
        if len(lines) != n:
            return None

        grid = []
        for line in lines:
            if len(line) != n:
                return None
            try:
                row = [int(char) for char in line]
            except ValueError:
                return None
            grid.append(row)

    return grid


def sudoku_score(prediction, ground_truth, strict=False):
    prediction = parse_sudoku(prediction, strict=strict)
    ground_truth = parse_sudoku(ground_truth, strict=True)

    return float(prediction == ground_truth)


parallel_bench_metric_func_map = {
    "list_match_score": list_match_score,
    "list_shuffle_score": list_shuffle_score,
    "list_random_insert_score": list_random_insert_score,
    "list_random_remove_score": list_random_remove_score,
    "list_random_replace_score": list_random_replace_score,
    "sentence_random_remove_score": sentence_random_remove_score,
    "sentence_random_replace_score": sentence_random_replace_score,
    "sentence_random_insert_score": sentence_random_insert_score,
    "sentence_replace_all_with_unique_random_score": sentence_replace_all_with_unique_random_score,
    "latin_square_score": latin_square_score,
    "math_op_score": math_op_score,
    "domino_score": domino_score,
    "sentence_to_words_score": sentence_to_words_score,
    "grammar_score": grammar_score,
    "regex_match_score": regex_match_score,
    "text_to_regex_score": text_to_regex_score,
    "json_syntax_score": json_syntax_score,
    "summary_score": SummaryScore,
    "paraphrase_score": ParaphraseScore,
    "startwith_score": startwith_score,
    "sudoku_score": sudoku_score,
}
