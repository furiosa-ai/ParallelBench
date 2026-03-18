# ParallelBench: Understanding the Tradeoffs of Parallel Decoding in Diffusion LLMs

<p align="center">
<img src = "docs/img/banner.png" width="70%" height="auto">
</p>

<p align="center">
      <a href="https://scholar.google.com/citations?user=Q-ARWkwAAAAJ&hl=eh" target="_blank">Wonjun Kang</a><sup>*1,5</sup>,
      <a href="https://scholar.google.com/citations?user=G1EpeWYAAAAJ&hl=en" target="_blank">Kevin Galim</a><sup>*1</sup>,
      <a href="https://scholar.google.com/citations?user=IXJcR1gAAAAJ&hl=en" target="_blank">Seunghyuk Oh</a><sup>*1</sup>,
      <a href="https://scholar.google.com/citations?user=XJXKp60AAAAJ&hl=en" target="_blank">Minjae Lee</a><sup>1</sup>,
      <a href="https://yzeng58.github.io/zyc_cv/" target="_blank">Yuchen Zeng</a><sup>2,3</sup>,
      <a href="https://scholar.google.com/citations?user=jkXzD7YAAAAJ&hl=en" target="_blank">Shuibai Zhang</a><sup>2</sup>,<br>
      <a href="https://scholar.google.com/citations?user=si-368wAAAAJ&hl=en" target="_blank">Coleman Hooper</a><sup>4</sup>,
      <a href="https://yuezhouhu.github.io/" target="_blank">Yuezhou Hu</a><sup>4</sup>,
      <a href="https://scholar.google.com/citations?user=Oyy8aDMAAAAJ&hl=en" target="_blank">Hyung Il Koo</a><sup>1</sup>,
      <a href="https://ece.snu.ac.kr/en/research-faculty/faculty/fulltime?md=view&profid=p041" target="_blank">Nam Ik Cho</a><sup>5</sup>,
      <a href="https://kangwooklee.com/aboutme/" target="_blank">Kangwook Lee</a><sup>2,6,7</sup>
  </p>
  <p  align="center">
    <sup>1</sup>FuriosaAI, <sup>2</sup>UW-Madison, <sup>3</sup>Microsoft Research, <sup>4</sup>UC Berkeley,<br>
    <sup>5</sup>Seoul National University, <sup>6</sup>KRAFTON, <sup>7</sup>Ludo Robotics
   </p>
<p align="center">
    <a href="https://parallelbench.github.io/"><img alt="Project" src="https://img.shields.io/static/v1?label=Project&message=Github&color=blue&logo=github-pages"></a>
    <a href="https://arxiv.org/abs/2510.04767"><img alt="arXiv" src="https://img.shields.io/badge/arXiv-2510.04767-b31b1b.svg"></a>
</p>


## 🔔 Updates

- **Jan 25, 2026** Paper accepted at ICLR 2026! 🎉
- **Oct 6, 2025** ParallelBench release!

## 🌍 Papers Using ParallelBench
The following works have evaluated their methods using ParallelBench. Check out how they tackle the speed-quality trade-off of parallel decoding!
- [Enabling Approximate Joint Sampling in Diffusion LMs](https://arxiv.org/abs/2509.22738)
- [Corrective Diffusion Language Models](https://arxiv.org/abs/2512.15596)

## 🗺️ Roadmap
We are currently working to support **new models** and implement **advanced unmasking methods**. If you are conducting dLLM research and would like to **contribute new models or methods**, please open an issue.

**New Models**
- [Fast-dLLM v2](https://github.com/NVlabs/Fast-dLLM)
- [LLaDA-MoE](https://github.com/ML-GSAI/LLaDA), [LLaDA2.x](https://github.com/inclusionAI/LLaDA2.X)

**Advanced Unmasking Methods**

- [WINO](https://github.com/Feng-Hong/WINO-DLLM?tab=readme-ov-file)
- [DUS](https://github.com/omerlux/DUS)
- [APD](https://github.com/danielmisrael/apd)
- [SlowFast Sampling](https://github.com/LiangrunFlora/Slow-Fast-Sampling)
- [EB-Sampler](https://arxiv.org/abs/2505.24857v1)
- [KLASS](https://github.com/shkim0116/KLASS)
- [Uncode](https://github.com/NEUIR/Uncode?tab=readme-ov-file) (formerly, PC-Sampler)

## 🔎 Overview
<p align="center">
<img src = "docs/img/teaser.png" width="100%" height="auto">
</p>

Diffusion LLMs (dLLMs) promise faster generation via parallel decoding. However, this speed often comes at the cost of quality, as they ignore token dependencies, an issue that existing benchmarks do not sufficiently capture. To address this issue, we introduce **ParallelBench**, the first benchmark designed to rigorously test this trade-off through realistic tasks that humans and autoregressive (AR) LLMs can easily solve, but which cause dLLMs to collapse as parallelism grows. We release **ParallelBench** to drive research towards truly efficient dLLMs that can overcome this challenge.

### Features

- **Information-Theoretic Analysis**: Error bounds on parallel decoding for tasks with inter-token dependencies, showing accuracy degradation as parallelism grows.
- **Quantitative Case Studies**: Synthetic list operations (Copy, Replace, Shuffle) with closed-form accuracy formulas that pin down where parallel decoding breaks.
- **17 Benchmark Tasks**: Three categories (Waiting Line, Text Writing, Puzzles) that humans and AR LLMs solve easily but expose quality drops in dLLMs under parallel decoding.


## 📐 Key Concepts

ParallelBench measures how **quality degrades as parallelism increases** in dLLMs. The key variable is **tokens per step (TPS)** — the number of tokens generated in parallel at each denoising step.

| Tokens per step | Meaning |
| :-: | --- |
| **1** | One-by-one decoding (equivalent to AR) |
| **k** | k tokens decoded in parallel per step |
| **max_tokens** | Fully parallel (one-step generation) |

ParallelBench evaluates **model + unmasking method** combinations. The same model can yield very different quality-speed trade-offs depending on which unmasking method is used.

The benchmark score is **PBx** — the maximum TPS at which a given combination still achieves at least **x%** average accuracy across all tasks. For example, PB80 = 8 means the combination can decode up to 8 tokens in parallel while maintaining ≥ 80% accuracy. Higher PBx values indicate better quality preservation under parallel decoding.

## ⚙️ Setup

### 1. Prerequisites
- **Conda**: For managing the environment.
- **NVIDIA GPU**: CUDA >= 11.8.
- **Java Development Kit (JDK)**: Required only for grammar-based evaluation metrics.

### 2. Set Python Environment

We use `uv` for faster package installation. The following commands will install PyTorch, `vLLM` for the LLM baselines, and all other required packages from `requirements.txt`.

```bash
# Use curl to download the script and execute it with sh:
curl -LsSf https://astral.sh/uv/install.sh | sh
# Install core dependencies
uv sync
```

### 3. Install Java (Optional)

If you need to run the **grammar-based** evaluations, install the JDK:

```bash
apt update
apt-get install openjdk-17-jdk -y
```

## ⚡ Quickstart

```bash
# Browse tasks (no GPU required)
pb browse                              # List all available tasks
pb browse waiting_line/copy            # View samples from a specific task
pb browse waiting_line/copy --index 3  # View a specific sample by index

# Run evaluation on a single task
pb eval --model parallelbench_llada \
  --model_args model_path=GSAI-ML/LLaDA-1.5 \
  --gen_kwargs k=32,unmasking=random \
  --tasks parallelbench_waiting_line_copy \
  --include_path parallelbench/tasks \
  --batch_size 1
```

## 🎯 Evaluation Coverage

### Tasks

| Category | Task | CLI task name |
| --- | --- | --- |
| Waiting Line (10) | Copy | `parallelbench_waiting_line_copy` |
| | Insert (index) | `parallelbench_waiting_line_insert_index` |
| | Insert (random) | `parallelbench_waiting_line_insert_random` |
| | Remove (index) | `parallelbench_waiting_line_remove_index` |
| | Remove (random) | `parallelbench_waiting_line_remove_random` |
| | Replace (index) | `parallelbench_waiting_line_replace_index` |
| | Replace (random) | `parallelbench_waiting_line_replace_random` |
| | Reverse | `parallelbench_waiting_line_reverse` |
| | Shuffle | `parallelbench_waiting_line_shuffle` |
| | Sort | `parallelbench_waiting_line_sort` |
| Text Writing (5) | Paraphrasing | `parallelbench_text_writing_paraphrasing` |
| | Summarization | `parallelbench_text_writing_summarization` |
| | Words to Sentence (easy) | `parallelbench_text_writing_words_to_sentence_easy` |
| | Words to Sentence (medium) | `parallelbench_text_writing_words_to_sentence_medium` |
| | Words to Sentence (hard) | `parallelbench_text_writing_words_to_sentence_hard` |
| Puzzles (2) | Latin Square (4x4) | `parallelbench_puzzles_latin_square_n4` |
| | Sudoku (4x4) | `parallelbench_puzzles_sudoku_n4` |

### Models

For additional models and unmasking methods, please refer to the [Roadmap](https://github.com/furiosa-ai/ParallelBench/#%EF%B8%8F-roadmap) section.

| Model family | CLI wrapper (`--model`) | Example `model_path` |
| --- | --- | --- |
| LLaDA | `parallelbench_llada` | `GSAI-ML/LLaDA-1.5` |
| Dream, DiffuCoder | `parallelbench_dream` | `Dream-org/Dream-v0-Instruct-7B` |
| SDAR, TraDo | `parallelbench_trado` | `JetAstra/SDAR-1.5-8B` |
| SEDD | `parallelbench_sedd` | `louaaron/sedd-medium` |
| AR baselines (vLLM) | `parallelbench_ar` | `meta-llama/Llama-3.1-8B-Instruct` |
| API models | `parallelbench_api` | Haiku, Mercury (requires `.env` keys) |

> **Adding your own model?** See the [step-by-step guide](docs/adding_custom_models.md) and the example in `parallelbench/models/local/example/`.

### Unmasking Methods

| Strategy | Type | CLI value | Description |
| -------- | ---- | --------- | ----------- |
| Random | Top-k (static) | `random` | Randomly selects which masked tokens to unmask |
| Origin | Top-k (static) | `origin` | Dream's native timestep-based unmasking (default for Dream models) |
| Confidence | Top-k (static) | `confidence_topk` | Unmasks tokens with highest model confidence |
| Margin | Top-k (static) | `topk_margin` | Unmasks tokens with largest margin between top-2 predictions |
| Entropy | Top-k (static) | `entropy_topk` | Unmasks tokens with lowest prediction entropy |
| Confidence Threshold | Adaptive | `confidence_threshold` | Unmasks all tokens above a confidence threshold (`alg_threshold`) |
| Confidence Factor | Adaptive | `confidence_factor` | Scales unmask count by a factor (`alg_factor`) |

**Top-k (static)** methods unmask a fixed number of tokens per step — tokens per step is constant.
**Adaptive** methods unmask a variable number of tokens per step — tokens per step varies, and the actual NFE (number of forward passes) is measured after generation.

> **Adding your own method?** See the [step-by-step guide](docs/adding_custom_unmasking_methods.md).

## 🚀 Running Evaluations

For the full CLI reference, generation parameters, and examples, see the [Running Evaluations guide](docs/running_evaluations.md).


## 🙏 Acknowledgements
Built on these open-source projects:

- [LLaDA](https://github.com/ML-GSAI/LLaDA)
- [Dream](https://github.com/DreamLM/Dream)
- [Fast-dLLM](https://github.com/NVlabs/Fast-dLLM)
- [Score-Entropy-Discrete-Diffusion](https://github.com/louaaron/Score-Entropy-Discrete-Diffusion)

## 📖 Citation
```bibtex
@article{kang2025parallelbench,
  title={ParallelBench: Understanding the Trade-offs of Parallel Decoding in Diffusion LLMs},
  author={Kang, Wonjun and Galim, Kevin and Oh, Seunghyuk and Lee, Minjae and Zeng, Yuchen and Zhang, Shuibai and Hooper, Coleman and Hu, Yuezhou and Koo, Hyung Il and Cho, Nam Ik and others},
  journal={arXiv preprint arXiv:2510.04767},
  year={2025}
}
```
