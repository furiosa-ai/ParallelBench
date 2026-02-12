set -eou pipefail

uv run run_all.py eval.py --device 0 1 --cfg cfg/debug/dream_default_list.yaml
uv run run_all.py eval.py --device 0 1 --cfg cfg/debug/dream_threshold_list.yaml
uv run run_all.py eval.py --device 0 1 --cfg cfg/debug/dream_factor_list.yaml


uv run run_all.py eval.py --device 0 1 --cfg cfg/debug/llada_1_5_default_list.yaml
uv run run_all.py eval.py --device 0 1 --cfg cfg/debug/llada_1_5_threshold_list.yaml
uv run run_all.py eval.py --device 0 1 --cfg cfg/debug/llada_1_5_factor_list.yaml

uv run run_all.py eval.py --device 0 1 --cfg cfg/debug/sedd_default_list.yaml
uv run eval.py --cfg cfg/debug/trado_default.yaml
uv run eval.py --cfg cfg/debug/trado_threshold.yaml


