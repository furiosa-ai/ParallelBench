# pytest configuration and fixtures

import shutil

import pytest

from parallelbench.datasets.task import create_parallelbench_task
from parallelbench.datasets.task_utils import _get_task_file, load_task_configs


# Tasks that tests depend on for local file loading
_TEST_TASKS_TO_GENERATE = [
    "waiting_line/copy",
    "waiting_line/shuffle",
]


@pytest.fixture(scope="session", autouse=True)
def generate_test_data():
    """Generate small JSONL + task_config.yaml files for tests that call load_task()."""
    configs = load_task_configs("test/waiting_line")
    generated_dirs = set()

    for task_name in _TEST_TASKS_TO_GENERATE:
        task_config = {
            **configs[task_name],
            "num_samples": 5,
            "samples_per_length": 0,
            "icl_example_count": 0,
        }
        create_parallelbench_task(split="test", task=task_config, output_file=None)
        generated_dirs.add(_get_task_file("test", task_name).parent)

    yield

    for directory in generated_dirs:
        if directory.exists():
            shutil.rmtree(directory)
