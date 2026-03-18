# Agent-Friendly CLI Implementation Plan

> Reference: [Rewrite Your CLI for AI Agents](https://justin.poehnelt.com/posts/rewrite-your-cli-for-ai-agents/) (Justin Poehnelt)

## Background

The blog post proposes 7 insights for making CLIs agent-friendly.
This plan evaluates each insight against ParallelBench's research benchmark context
and defines a focused implementation roadmap.

## Insight Applicability Assessment

| #   | Insight                                 | Value  | Verdict                                                          |
| --- | --------------------------------------- | ------ | ---------------------------------------------------------------- |
| 1   | Raw JSON output                         | HIGH   | Implement — agents need to parse eval results and browse samples |
| 2   | Runtime schema introspection            | LOW    | Skip — 4 commands, `--help` + CLAUDE.md is sufficient            |
| 3   | Context window discipline (field masks) | MEDIUM | Implement (lightweight) — `--fields` for `analyze`               |
| 4   | Hardened input validation               | MEDIUM | Partial — already decent via argparse                            |
| 5   | Agent skills as first-class docs        | LOW    | Skip — CLAUDE.md already serves this role                        |
| 6   | Multi-surface exposure (CLI + MCP)      | LOW    | Skip — no MCP server needed for a research tool                  |
| 7   | Safety rails (dry-run)                  | MEDIUM | Partial — extend `--dry_run` to `eval`                           |

## Principles

1. **Agent-parseable output is the highest-value change** — `--output_format json` across commands unlocks programmatic consumption
2. **Minimal disruption to existing UX** — default behavior (Rich tables) must not change
3. **Incremental adoption** — each change is independently shippable and useful
4. **No over-engineering** — skip MCP, schema introspection, YAML skill docs
5. **Test coverage parity** — every new CLI feature gets tests matching existing patterns

## Decision Drivers

1. **Agent automation readiness** — agents calling `pb analyze results/ --output_format json` is the primary use case
2. **Implementation cost vs. value** — 4 commands, ~1200 LOC total; changes must be proportional
3. **Backward compatibility** — defaults stay Rich, new flags are opt-in

## Chosen Option: Focused JSON Output Layer

### Why not a full rewrite?

- `eval` delegates entirely to lm-eval's `cli_evaluate()` — full abstraction is impractical without forking lm-eval
- 4 commands don't justify a framework migration (e.g., Typer + JSON middleware)
- Schema introspection adds complexity for a use case that CLAUDE.md already solves
- Existing data pipelines (`list[dict]`) already produce structured data — only a thin serialization layer is needed

---

## Implementation Roadmap

### PR 1: Shared Output Utility + `analyze --output_format json` + `--fields`

**Priority:** HIGH
**Files:**

| File                                   | Change                                                                                                    |
| -------------------------------------- | --------------------------------------------------------------------------------------------------------- |
| `parallelbench/cli/output.py` (new)    | Shared `--output_format` argparse argument, `Console(stderr=True)` manager, `write_json_output()` helper  |
| `parallelbench/cli/analyze.py`         | Add `--output_format json` support, add `--fields` filtering                                              |
| `tests/test_analyze_cli.py` (new)      | JSON output validation, fields filtering validation                                                       |

**Why `analyze` first:**
- `_collect_rows()` already returns `list[dict]` (line 118-156)
- `_export_csv()` (line 283-288) is a direct precedent for structured output
- `compute_pb_scores()` (in `pb_score.py`) returns a clean `dict[str, float | None]`
- This is the command where agents benefit most (parsing evaluation results)

**JSON envelope schema:**

```json
{
  "rows": [
    {
      "model": "LLaDA-1.5",
      "task": "waiting_line/copy",
      "unmasking": "low_confidence",
      "k": 4,
      "metrics": {
        "accuracy": 0.95,
        "nfe": 32
      }
    }
  ],
  "pb_scores": {
    "PB90": 0.85,
    "PB80": 0.72
  },
  "metadata": {
    "n_rows": 42,
    "output_path": "results/"
  }
}
```

**`--fields` behavior:**
- Filter which columns appear in output (both Rich table and JSON)
- Example: `pb analyze results/ --fields model,task,accuracy`
- Built on existing `DISPLAY_COLUMNS` / `CSV_COLUMNS` lists (line 50-65)

**Acceptance criteria:**
- `pb analyze results/ --output_format json | python -m json.tool` produces valid JSON
- `pb analyze results/ --output_format json --fields model,task | jq '.rows[0] | keys'` returns only specified fields
- `pb analyze results/` (no flag) behaves identically to current behavior
- Rich status/progress output goes to stderr when `--output_format json`

**Shared `output.py` utility design:**
- `add_output_format_argument(parser)` — adds `--output_format` to any argparse parser
- `create_console(output_format)` — returns `Console(stderr=True)` when JSON, normal otherwise
- `write_json_output(data)` — `json.dumps()` to stdout with consistent formatting

---

### PR 2: `browse --output_format json`

**Priority:** MEDIUM
**Files:**

| File                          | Change                                                       |
| ----------------------------- | ------------------------------------------------------------ |
| `parallelbench/cli/browse.py` | Add `--output_format json`, redirect Rich spinners to stderr |
| `tests/test_browse_cli.py`    | Add JSON output tests                                        |

**Key implementation note:**
`browse.py` uses `console.status()` (lines 253, 260) for Rich spinners on stdout.
When `--output_format json`, these must go to stderr to avoid corrupting JSON output.
Use the shared `create_console()` from `output.py` (established in PR 1).

**JSON schema — task list mode (`pb browse --output_format json`):**

```json
{
  "tasks": [
    {
      "name": "waiting_line/copy",
      "category": "waiting_line",
      "n_samples": 50,
      "description": "..."
    }
  ]
}
```

**JSON schema — sample mode (`pb browse waiting_line/copy --output_format json`):**

```json
{
  "task": "waiting_line/copy",
  "samples": [
    {
      "index": 0,
      "input": { "messages": [...] },
      "label": "...",
      "metadata": { ... }
    }
  ]
}
```

**Acceptance criteria:**
- `pb browse --output_format json | jq '.tasks | length'` returns task count
- `pb browse waiting_line/copy --output_format json --index 0 | jq '.samples[0].input'` returns sample input
- Spinners/status text never appears on stdout when `--output_format json`

---

### PR 3: `eval --dry_run`

**Priority:** MEDIUM (implement when a concrete agent workflow needs it)
**Files:**

| File                            | Change                                                   |
| ------------------------------- | -------------------------------------------------------- |
| `parallelbench/cli/eval.py`     | Add `--dry_run` flag, intercept before `cli_evaluate()`  |
| `tests/test_eval_cli.py` (new)  | Dry run validation                                       |

**Scope limitation:**
`eval.py` delegates to lm-eval's `cli_evaluate()` (line 55), making full config resolution
impractical. Scope `--dry_run` to ParallelBench-specific validation only:

1. Parse `--model_args` (comma-separated key=value)
2. Validate model wrapper can be instantiated (registry lookup)
3. Resolve generation config via `DLLMBase._build_generation_config()`
4. Print resolved config as JSON and exit — no GPU usage

**What `--dry_run` does NOT validate:**
- lm-eval's internal task loading or argument parsing
- Actual model weight loading or GPU allocation
- Task YAML resolution

**Acceptance criteria:**
- `pb eval --model parallelbench_llada --model_args model_path=X,steps=32 --dry_run` prints resolved config, exits 0
- Invalid generation config (e.g., `steps=7,max_tokens=32,block_length=8`) prints validation error, exits 1
- No GPU memory allocated during dry run

---

## Skipped Items

| Item                          | Reason                                                            |
| ----------------------------- | ----------------------------------------------------------------- |
| MCP server                    | Over-engineering for 4 commands in a research tool                |
| Runtime schema introspection  | `--help` and CLAUDE.md are sufficient for discovery               |
| YAML agent skill documents    | CLAUDE.md already provides agent-specific guidance                |
| JSON input (`--json '{...}'`) | `eval` delegates to lm-eval CLI — cannot intercept input parsing  |
| `data --output_format json`   | Already outputs JSONL — effectively JSON                          |
| Structured error output       | Low priority; can add later if JSON consumers need it             |

## Architecture Notes

### stderr/stdout Separation

When `--output_format json` is active:
- **stdout**: only the JSON payload
- **stderr**: Rich progress spinners, warnings, status messages

This follows the existing pattern in `analyze.py:24` (`console_err = Console(stderr=True)`).

### No Breaking Changes

All changes are additive — new optional flags with no effect on default behavior.
Existing scripts and workflows continue to work unchanged.

### Antithesis Considered

> "Agent-parseable CLI output is solving the wrong problem.
> Researchers run experiments, not CI pipelines.
> The bottleneck is GPU inference time, not CLI output parsing.
> Results are already JSON files on disk."

This is valid. The plan mitigates by keeping implementation cost minimal (thin serialization layer
on existing `list[dict]` pipelines). Even without agent consumers, `--output_format json`
benefits scripting and piping workflows (e.g., `pb analyze results/ --output_format json | jq`).
