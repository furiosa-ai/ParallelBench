# ParallelBench Dataset Data

Minimal dataset specs for generation-first workflow.

This directory intentionally keeps only the inputs needed to generate datasets
from scratch and push to Hugging Face without writing local JSONL outputs.

Included:
- `task_configs/`: category-first generation specs
  - `test/waiting_line.yaml`
  - `test/waiting_line_n15.yaml`
  - `test/text_writing.yaml`
  - `test/puzzles.yaml`
  - `train/waiting_line.yaml`
  - `train/waiting_line_n15.yaml`
  - `train/text_writing.yaml`
  - `train/puzzles.yaml`
- `resources/`: generation resources used by current configs
  - `first_last_names.yaml`

Excluded on purpose:
- `output/`: pre-generated JSONL datasets
- `_old/`: legacy artifacts
- `src/`, `llm_generated/`: generation sources not required by current task generators

Compatibility note:
- `load_words_from_file()` resolves `first_last_names.yaml` and
  older legacy paths by searching:
  `data/`, `data/resources/`, and legacy paths
  `data/resources/lexicons/`, `data/resources/names/`.

Design note:
- Task configs are grouped by HF upload category instead of legacy
  zero-shot/one-shot naming conventions.
- Train configs are derived from test task definitions via `tasks_from`,
  so test remains the source-of-truth for task schema/prompts.
