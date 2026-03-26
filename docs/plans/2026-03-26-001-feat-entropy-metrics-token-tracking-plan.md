---
title: "feat: Entropy metrics and token usage tracking per generation"
type: feat
status: completed
date: 2026-03-26
---

# feat: Entropy metrics and token usage tracking per generation

## Overview

Add per-generation measurement of Structural Entropy, Coupling Entropy, and token usage (input, output, total) to the HyperAgents evolutionary loop, storing results in each generation's `metadata.json` and in `archive.jsonl` entries. Implements the preliminary data collection described in *Entropic Selection for Self-Improving Agents* (Stanford, 2026) — metrics only, no changes to the selection algorithm.

## Problem Frame

The paper proposes entropy-based fitness criteria for parent selection in evolutionary agent frameworks. Before implementing entropic selection, we need empirical data: how do Structural Entropy and Coupling Entropy evolve across generations? What is the relationship between these static metrics and Modification Cost (token spend)?

Currently, HyperAgents records no token usage and computes no codebase structure metrics. This plan adds both, enabling retrospective and prospective analysis of entropy trajectories.

## Requirements Trace

- R1. Track token usage (input tokens, output tokens, total) for each meta agent invocation and store per generation
- R2. Compute Structural Entropy H_struct per generation per the paper's Eq. 2
- R3. Compute Coupling Entropy H_couple per generation per the paper's Eq. 4
- R4. Store all metrics in `metadata.json` (per-generation) and as a `metrics` field in `archive.jsonl` entries
- R5. Works with Anthropic API key and OpenAI API key providers (litellm-compatible token extraction)
- R6. Provide a backfill script to compute entropy metrics on existing output directories (token usage cannot be backfilled)
- R7. No changes to the selection algorithm, fitness function, or parent selection logic

## Scope Boundaries

- No implementation of the entropic fitness function (Eq. 7) or modified parent selection
- No Modification Cost normalization or cross-run scaling (raw C_mod values only)
- No hierarchical entropy (Eq. 3) in this phase — flat H_struct only (deferred to next phase)
- Token tracking covers the meta agent only; task agent token usage is not tracked in this phase
- Coupling graph covers Python module-level import relationships only (no function-level call graphs)
- Non-Python files (markdown, CSV, JSON data) excluded from both entropy computations
- No changes to how Docker containers are built or run

## Context & Research

### Formal Definitions (from the paper)

**Structural Entropy** (Eq. 2): Shannon entropy of the file-size distribution across the repository.

```
H_struct(R) = -Σ p_i log2(p_i)    where p_i = |f_i| / |R|
```

`|f_i|` = size of file `f_i` (character count as proxy for tokens); `|R|` = total size. Files with zero size excluded. Applied over all `.py` files in the agent codebase (excluding `outputs/`, `domains/data`, test files).

**Coupling Entropy** (Eq. 4): Shannon entropy of the normalized fan-out distribution in the import dependency graph.

```
H_couple(G) = -Σ r_i log2(r_i)    where r_i = fan-out(v_i) / Σ fan-out(v_j)
```

`fan-out(v_i)` = number of local repo modules that file `v_i` imports. Modules with fan-out = 0 contribute r_i = 0 (excluded from entropy sum, per convention). `G = (V, E)` built from Python `import` and `from X import Y` statements; only edges to files within the repo count.

**Modification Cost** (Eq. 5): Total tokens consumed by the meta agent in one self-modification step.

```
C_mod = T_input + T_output
```

Tracked as three values: `input_tokens`, `output_tokens`, `total_tokens`.

### Relevant Code and Patterns

- `agent/llm.py:get_response_from_llm()` — returns `(text, history, {})`. The third return value is an empty dict; token usage goes here.
- `agent/llm_withtools.py:chat_with_agent()` — calls `get_response_from_llm()` in a loop; accumulates calls across tool turns. Token totals must be summed across all turns.
- `meta_agent.py:MetaAgent.forward()` — calls `chat_with_agent()`; indirectly the accumulation point.
- `run_meta_agent.py` — the entry point run inside Docker; writes `model_patch.diff` to `--outdir`. Token usage JSON should be written here too.
- `utils/gl_utils.py:update_and_save_archive()` — appends one JSON line per generation to `archive.jsonl`. Extend to accept optional `metrics` dict.
- `utils/gl_utils.py:update_node_metadata()` — updates `metadata.json` for a genid. Use this to store metrics after compute.
- `generate_loop.py:generate()` — finalization block (lines 691–715) writes metadata. Compute entropy here after the container's `agent_output/` is copied to host.
- `analysis/plot_progress.py` — existing analysis module; place new entropy module alongside it.

### Token Extraction — Provider Compatibility

litellm normalizes usage across providers. The `response` object from `litellm.completion()` has:
- `response.usage.prompt_tokens` (OpenAI, Anthropic)
- `response.usage.completion_tokens`
- `response.usage.total_tokens`

These fields are present when using `ANTHROPIC_API_KEY` (direct Anthropic), `OPENAI_API_KEY`, and `LLM_PROVIDER=oauth` (ccproxy forwards standard usage fields). litellm's `drop_params=True` is already set; usage extraction is unaffected.

### Reconstruction Approach for Entropy Computation

The meta agent runs inside Docker and leaves the host repo in base-commit state after each generation. `model_patch.diff` is the cumulative diff from base_commit representing the full agent codebase at generation N (including all parent lineage patches, per `diff_versus_commit(base_commit)`).

Entropy is computed on the host by:
1. Copying the repo to a temp directory (`tempfile.mkdtemp`)
2. Applying `model_patch.diff` with `git apply` (or `patch -p1`)
3. Analyzing the resulting file tree
4. Deleting the temp directory

For generations with empty patches (agent made no changes), entropy equals the baseline repo's entropy.

### Institutional Learnings

- No `docs/solutions/` entries; this is new analytical infrastructure.

## Key Technical Decisions

- **tiktoken for file token counts**: Use `tiktoken.get_encoding("cl100k_base").encode(file_content)` to measure file size as actual token count. `cl100k_base` is the encoding used by GPT-3.5/GPT-4 and serves as a consistent, provider-agnostic measure regardless of which LLM is active. tiktoken is already a transitive dependency of litellm but must be explicitly pinned in `requirements.txt` to ensure it is reliably available in the host venv and inside the Docker image. A Docker image rebuild is required after adding tiktoken to `requirements.txt`.

- **Token accumulation in `chat_with_agent()`**: `chat_with_agent()` is the natural accumulation boundary — it spans the full meta agent interaction. It already returns `new_msg_history`; extend to also return a `usage` dict. `run_meta_agent.py` writes this to `agent_output/token_usage.json`. Host reads it after the standard `copy_from_container` call.

- **Storage in both metadata.json and archive.jsonl**: `metadata.json` (per genid) is the authoritative record; `archive.jsonl` entries gain a `metrics` key for convenient streaming reads without needing to open per-generation directories. `update_and_save_archive()` accepts an optional `metrics` kwarg.

- **Coupling graph scope — local imports only**: Only edges where the imported module resolves to a `.py` file within the repo root count. stdlib and third-party imports (`litellm`, `backoff`, etc.) are excluded. This makes H_couple purely a measure of internal structure rather than external dependency footprint.

- **Files included in entropy analysis**: All `.py` files reachable from repo root, excluding:
  - `outputs/` (generated artifacts)
  - `venv_nat/`, `.venv/`, `venv/` (virtual envs)
  - `__pycache__/` (compiled bytecode)
  - `tests/` (deferred — including tests shifts entropy toward infrastructure rather than agent logic; revisit for paper experiments)
  - `docs/` (no Python files)

- **Zero fan-out nodes**: Files that import nothing locally (e.g., `utils/constants.py`) have fan-out = 0. Excluded from entropy sum (0 log 0 = 0 by convention). This is consistent with the paper's formulation.

- **Graceful degradation**: If entropy computation fails (e.g., diff fails to apply, AST parse error), log a warning, store `null` for that metric, and do not crash the generation loop.

## Open Questions

### Resolved During Planning

- **Where does token accumulation happen if meta agent errors out?** `run_meta_agent.py` writes `token_usage.json` in a `finally` block — always written, even on crash, with whatever was accumulated before the error.
- **Can H_couple be computed if there are no imports?** Yes — all fan-outs are 0, entropy is 0 by convention.
- **Do we need to rebuild the Docker image to get token tracking?** No — since the repo is volume-mounted, code changes in `agent/llm.py`, `agent/llm_withtools.py`, and `run_meta_agent.py` take effect immediately in the next container run.

### Deferred to Implementation

- **tiktoken encoding choice**: `cl100k_base` is used throughout. If a future provider uses a materially different vocabulary, the encoder can be swapped — the entropy computation is unaffected as long as it's consistent across all files in a given run.
- **Hierarchical entropy (H_hier, Eq. 3)**: Deferred. Requires defining depth weight scheme ($w_k$). Add to `entropy_metrics.py` as a stub with a TODO.
- **Which files to include in entropy scope is revisitable**: The exclusion of `tests/` may warrant reconsideration for the paper's full experiments. Documented in the module so it's easy to adjust.

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
generate_loop.py: generate()
│
├─ [inside Docker container]
│   ├─ run_meta_agent.py
│   │   ├─ chat_with_agent() ← accumulates token usage across turns
│   │   └─ writes agent_output/token_usage.json (always, via finally)
│   └─ writes agent_output/model_patch.diff
│
├─ copy_from_container("/tmp/agent_output", local_agentoutput_folder)
│
├─ [host-side, after copy]
│   ├─ read token_usage.json → {input_tokens, output_tokens, total_tokens}
│   ├─ reconstruct_repo_at_gen(repo_path, patch_file) → temp_dir
│   ├─ compute_structural_entropy(temp_dir) → H_struct
│   ├─ compute_coupling_entropy(temp_dir) → H_couple
│   └─ cleanup temp_dir
│
├─ update_node_metadata(output_dir, genid, {"metrics": {...}})
└─ update_and_save_archive(..., metrics={...})  ← archive.jsonl entry gains metrics
```

**archive.jsonl entry (extended):**
```json
{
  "current_genid": 3,
  "archive": ["initial", 1, 2, 3],
  "metrics": {
    "h_struct": 2.847,
    "h_couple": 1.923,
    "c_mod": {"input_tokens": 14200, "output_tokens": 3100, "total_tokens": 17300},
    "repo_file_count": 18,
    "repo_total_tokens": 21080
  }
}
```

## Implementation Units

- [x] **Unit 1: Token usage extraction in llm.py and llm_withtools.py**

**Goal:** Instrument `get_response_from_llm()` to return token usage from the litellm response, and aggregate total usage across all turns in `chat_with_agent()`.

**Requirements:** R1, R5

**Dependencies:** None

**Files:**
- Modify: `agent/llm.py`
- Modify: `agent/llm_withtools.py`
- Test: `tests/test_llm.py`

**Approach:**
- In `get_response_from_llm()`, extract `response.usage` (prompt_tokens, completion_tokens, total_tokens) from the litellm response object. Return as the third element of the tuple: `{"input_tokens": N, "output_tokens": N, "total_tokens": N}`. Gracefully handle missing usage fields (some providers may omit them) by defaulting to 0.
- In `chat_with_agent()`, initialize an accumulator dict `{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}`. After each `get_response_fn()` call, add the returned usage to the accumulator. Return the accumulator as a new third return value from `chat_with_agent()`.
- Update all call sites of `chat_with_agent()` that unpack the return value (meta_agent.py, task_agent.py) to handle the new return shape.

**Patterns to follow:**
- `agent/llm.py` return signature: `(str, list, dict)` — the third element already exists as `{}`; fill it in.
- Defensive coding: wrap usage extraction in try/except; log and return zeros on failure.

**Test scenarios:**
- `get_response_from_llm()` with a mock litellm response that includes usage fields → third return value contains correct token counts
- `get_response_from_llm()` with a response missing usage fields → returns `{"input_tokens": 0, "output_tokens": 0, "total_tokens": 0}` without raising
- `chat_with_agent()` mock with two turns → accumulated totals are the sum of both turns

**Verification:**
- `agent/llm.py:get_response_from_llm()` third return value is never an empty dict when the provider returns usage
- Existing `tests/test_llm.py` still passes
- `chat_with_agent()` returns a three-element tuple

---

- [x] **Unit 2: Token usage persistence in run_meta_agent.py**

**Goal:** Write the accumulated token usage from the meta agent's full run to `agent_output/token_usage.json` inside the container, so the host can read it after the standard `copy_from_container` call.

**Requirements:** R1, R4

**Dependencies:** Unit 1

**Files:**
- Modify: `run_meta_agent.py`
- Test: `tests/test_llm.py` (integration assertion)

**Approach:**
- `meta_agent.py:MetaAgent.forward()` currently calls `chat_with_agent()` and discards the return value. Capture the (now) third return value (accumulated usage dict).
- `run_meta_agent.py` already has the `--outdir` argument. In a `finally` block (so it runs even on agent error), write the usage dict to `os.path.join(args.outdir, "token_usage.json")`. If usage was never set (e.g., error before any LLM call), write zeros.
- The host reads this file from `local_agentoutput_folder/token_usage.json` after `copy_from_container`.

**Patterns to follow:**
- `run_meta_agent.py` already writes `model_patch_outfile` with `open(model_patch_outfile, "w")`; mirror this pattern for `token_usage.json`.

**Test scenarios:**
- Successful meta agent run → `token_usage.json` exists and contains non-zero tokens
- Meta agent errors mid-run → `token_usage.json` still written (possibly zeros)
- `token_usage.json` is valid JSON with keys `input_tokens`, `output_tokens`, `total_tokens`

**Verification:**
- `agent_output/token_usage.json` is present in the output directory after any meta agent run
- File is always parseable as JSON

---

- [x] **Unit 3: Entropy metrics computation module**

**Goal:** Implement `compute_structural_entropy()` and `compute_coupling_entropy()` as a standalone module with no dependencies beyond Python stdlib and math.

**Requirements:** R2, R3

**Dependencies:** None (standalone)

**Files:**
- Create: `analysis/entropy_metrics.py`
- Modify: `requirements.txt` (add `tiktoken>=0.7.0`)
- Test: `tests/test_entropy_metrics.py`

**Approach:**

*Structural Entropy:*
- Walk the repo path, collecting all `.py` files (excluding `outputs/`, venv dirs, `__pycache__/`, `tests/`; configurable via an exclusion list parameter).
- Initialize a `tiktoken` encoder once using `tiktoken.get_encoding("cl100k_base")`. For each file, record `size_i = len(encoder.encode(file_content))` (token count). If a file cannot be encoded (e.g., encoding error), fall back to `len(file_content) // 4` and log a warning.
- Compute `total = sum(size_i)`. If total == 0, return 0.
- Compute `p_i = size_i / total` for each file. Compute `H = -sum(p_i * log2(p_i) for p_i > 0)`.
- Return `{"h_struct": H, "file_count": m, "total_tokens": total}`.

*Coupling Entropy:*
- Walk same `.py` files to build a module name → file path index (e.g., `agent.llm` → `agent/llm.py`).
- For each file, parse with `ast.parse()`. Collect all `import X` and `from X import Y` statements. For each, attempt to resolve to a local repo file using the module index. Count resolved local imports as fan-out edges.
- Compute `fan_out[v]` for each file. If all fan-outs are 0, return `{"h_couple": 0, "node_count": N, "edge_count": 0}`.
- Normalize: `r_i = fan_out[v_i] / sum(fan_out)`. Compute `H = -sum(r_i * log2(r_i) for r_i > 0)`.
- Return `{"h_couple": H, "node_count": |V|, "edge_count": total fan-out}`.

*Repo reconstruction helper:*
- `reconstruct_and_analyze(repo_root, patch_file_path)` — copies repo to a temp dir, applies patch with subprocess `git apply`, runs both entropy functions, cleans up. Returns combined metrics dict. If patch file is empty or missing, run entropy functions on the unpatched repo (baseline).

**Patterns to follow:**
- `analysis/plot_progress.py` for module style and placement.
- Python stdlib `ast`, `math`, `pathlib`, `tempfile`, `subprocess` only — no new dependencies.

**Test scenarios:**
- Single-file repo → H_struct = 0 (all mass in one file), H_couple depends on imports
- Two equal-sized files → H_struct = 1.0 (maximum for m=2)
- File with no local imports → fan-out = 0, excluded from H_couple sum
- All files import each other equally → H_couple = log2(N)
- AST parse failure on one file → logged warning, file skipped, entropy computed on remaining files
- Empty patch file → returns baseline repo entropy (no reconstruction attempted)
- Valid patch file → temp dir is created, analyzed, and cleaned up regardless of success/failure

**Verification:**
- `tests/test_entropy_metrics.py` passes with synthetic file trees covering all above scenarios
- H_struct for a single large file = 0.0
- H_struct for N equal files = log2(N)
- H_couple for a chain of imports (A→B→C) has lower entropy than a star pattern (A→B, A→C, A→D)
- No temp directories left behind after any execution path

---

- [x] **Unit 4: Integration in generate_loop.py and gl_utils.py**

**Goal:** After each generation's agent output is copied to the host, compute entropy metrics and token usage, store in `metadata.json`, and include in the `archive.jsonl` entry.

**Requirements:** R1, R2, R3, R4

**Dependencies:** Units 1, 2, 3

**Files:**
- Modify: `generate_loop.py`
- Modify: `utils/gl_utils.py`
- Test: `tests/test_entropy_metrics.py` (integration scenario)

**Approach:**

*In `utils/gl_utils.py`:*
- Extend `update_and_save_archive(output_dir, archive, new_node, metrics=None)` to accept an optional `metrics` dict. When provided, include it in the JSONL entry: `{"current_genid": new_node, "archive": archive, "metrics": metrics}`. When absent, behavior is unchanged (backward compatible).

*In `generate_loop.py:generate()`:*
- After the `copy_from_container(container, "/tmp/agent_output", local_agentoutput_folder)` line (currently around line 612), add:
  1. Read `token_usage.json` from `local_agentoutput_folder` if present; default to zeros if missing.
  2. Call `reconstruct_and_analyze(repo_path, local_patch_file)` to get `{"h_struct": ..., "h_couple": ..., ...}`.
  3. Compose `metrics_dict` combining token usage and entropy values.
  4. Call `update_node_metadata(output_dir, current_genid, {"metrics": metrics_dict})` to persist in `metadata.json`.
- In the generation loop in `generate_loop()` (around line 926 where `update_and_save_archive` is called), pass the generation's computed metrics: `update_and_save_archive(output_dir, archive, current_genid, metrics=gen_metrics)`.
- Wrap all metric computation in try/except; log warnings on failure; store `null` values so generation never fails due to metric errors.

**Patterns to follow:**
- `utils/gl_utils.py:update_node_metadata()` for metadata write pattern.
- `generate_loop.py`'s existing try/except around container operations for graceful degradation.
- The existing pattern of reading `local_agentoutput_folder` files (e.g., `local_patch_file` check) for token_usage.json read.

**Test scenarios:**
- End-to-end: mock generate() flow → both metadata.json and archive.jsonl contain `metrics` key
- Entropy computation failure (bad patch) → metrics stored as nulls, generation continues
- Missing token_usage.json (agent crashed before writing) → token fields zero, entropy still computed
- `update_and_save_archive` called without `metrics` kwarg → archive.jsonl unchanged from current format

**Verification:**
- After running `generate_loop.py` for 2+ generations, each gen's `metadata.json` has a `metrics` key
- `archive.jsonl` entries (except "initial") have a `metrics` key with `h_struct`, `h_couple`, and `c_mod` sub-keys
- No generation fails due to metric computation errors

---

- [x] **Unit 5: Backfill script for existing output directories**

**Goal:** Provide `analysis/compute_entropy_backfill.py` to retroactively compute entropy metrics for all generations in an existing output directory (token usage cannot be backfilled).

**Requirements:** R6

**Dependencies:** Unit 3

**Files:**
- Create: `analysis/compute_entropy_backfill.py`

**Approach:**
- Accept `--output-dir <path>` as a CLI argument pointing to a run directory (e.g., `outputs/generate_20260325_212836_252984/`).
- For each generation found (by listing `gen_*/` subdirectories), read `agent_output/model_patch.diff` if it exists.
- Call `reconstruct_and_analyze(repo_root, patch_file)` for each generation.
- Write metrics to each gen's `metadata.json` via `update_node_metadata()`. Do not overwrite existing `metrics` keys unless `--force` flag is set.
- Print a summary table: genid | H_struct | H_couple | patch_lines | files_changed.
- Does not update `archive.jsonl` (archive entries are append-only; backfill only touches metadata.json).

**Patterns to follow:**
- `domains/report.py` or `analysis/plot_progress.py` CLI style for argparse.

**Test scenarios:**
- Run on a directory with 3 gens, 2 of which have non-empty patches → 3 metadata.json files updated
- Run on a gen with an empty patch → entropy equals baseline, stored without error
- Run with `--force` flag → existing metrics overwritten
- Run without `--force` on already-processed directory → existing metrics preserved, script exits gracefully

**Verification:**
- After running on `generate_20260325_212836_252984/`, gen_3 and gen_4 metadata.json files have a `metrics` key with correct entropy values
- Baseline gen (`gen_initial`) is handled gracefully (no patch file)

---

## System-Wide Impact

- **Interaction graph:** `generate_loop.py → gl_utils.update_and_save_archive` (modified signature, backward compatible). `run_meta_agent.py → meta_agent.MetaAgent.forward() → chat_with_agent()` (return signature changes). All callers of `chat_with_agent()` must be updated to handle three-tuple return.
- **Error propagation:** All metric computation is wrapped in try/except. A broken entropy computation produces null values in metadata; it does not propagate as an exception into the generation loop.
- **State lifecycle risks:** The temp directory created for codebase reconstruction must always be cleaned up. Use `shutil.rmtree` in a `finally` block. If the host repo is modified (volume mount), the patch application must target the TEMP copy only — never the live repo.
- **API surface parity:** `chat_with_agent()` return signature changes from `(str, list)` to `(str, list, dict)`. All callers — `meta_agent.py` and `task_agent.py` — must be updated. Check for any other callers in the repo.
- **Integration coverage:** Token tracking is end-to-end only verifiable with a live LLM call; unit tests should mock litellm and verify the plumbing. An integration test confirming real token values from Anthropic or OpenAI is useful but may be skipped in CI.

## Risks & Dependencies

- **`chat_with_agent()` return shape change** is a breaking change to internal callers. Must audit all call sites before merging.
- **Git apply in temp dir** requires the repo to be a valid git repository at the temp location. Since we copy the full repo, git history is included, and `git apply` should work. If the patch was generated with a different base commit (stale runs), `git apply` may fail — handle with `--reject` flag and log.
- **AST parse failures**: Some generated code may not be valid Python (agent wrote a broken file). Log and skip; do not crash entropy computation.
- **litellm usage field availability**: Usage fields are present for Anthropic and OpenAI. Other providers (Ollama) may not return usage. Handle missing fields as zeros.

## Documentation / Operational Notes

- The `metrics` key in `metadata.json` and `archive.jsonl` uses `null` (JSON) for fields that could not be computed, preserving type consistency for downstream analysis.
- File token counts use `tiktoken` with `cl100k_base` encoding. This is documented in `analysis/entropy_metrics.py` and should be cited in the paper's experimental description.
- Adding `tiktoken` to `requirements.txt` requires a Docker image rebuild (`--force-rebuild` flag in `generate_loop.py` or equivalent).
- To visualize entropy trajectories post-run, read `archive.jsonl` and extract `metrics.h_struct` and `metrics.h_couple` per `current_genid`.

## Sources & References

- Origin paper: `/Users/williamstanford/papers/Entropic Selection.pdf` — Eq. 2 (H_struct), Eq. 4 (H_couple), Eq. 5 (C_mod)
- Related code: `agent/llm.py`, `agent/llm_withtools.py`, `utils/gl_utils.py`, `generate_loop.py`, `run_meta_agent.py`
- litellm usage extraction: `response.usage.prompt_tokens`, `response.usage.completion_tokens`
- Python `ast` module docs for import parsing
