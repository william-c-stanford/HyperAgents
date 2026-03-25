---
title: "feat: Add Ollama provider support (Qwen2.5-Coder 7B)"
type: feat
status: active
date: 2026-03-25
---

# feat: Add Ollama provider support (Qwen2.5-Coder 7B)

## Overview

Add Ollama as a third LLM provider alongside Anthropic (OAuth/ccproxy) and OpenAI. The
primary target model is Qwen2.5-Coder 7B in a Metal-optimized GGUF quantization for Apple
M4, but the model tag is fully user-configurable via environment variable. Integration goes
through litellm's existing `ollama_chat/` prefix — no new library dependencies.

## Problem Frame

Users with a local Ollama installation should be able to run Hyperagents experiments against
a local model without any API key or cloud subscription. The M4 MacBook is the primary local
hardware; the default quantization should be tuned accordingly (`q4_K_M`). The integration
must be testable in both a local pytest session and inside the Docker container (via
`--network=host` forwarding to the host Ollama process).

## Requirements Trace

- R1. Users can select an Ollama model via `OLLAMA_MODEL` env var; a sane M4-optimized default is provided
- R2. `OLLAMA_API_BASE` env var overrides the Ollama host URL (default `http://localhost:11434`)
- R3. Integration routes through `litellm`'s `ollama_chat/` provider — no code changes to `get_response_from_llm` call sites
- R4. `agent/llm.py` exports an `OLLAMA_MODEL` constant following the existing `PROVIDER_MODEL` naming convention
- R5. pytest integration tests skip automatically when Ollama is not reachable
- R6. Docker container tests work when Ollama is running on the host (`--network=host`)
- R7. Docker image build tests continue to pass (Ollama integration tests excluded from build-time pytest step)
- R8. README documents how to pull the model and run tests

## Scope Boundaries

- **Out of scope:** Ollama model auto-pull during Docker build (model is ~4.7 GB; pull is a runtime concern)
- **Out of scope:** streaming support — `get_response_from_llm` is not streaming today
- **Out of scope:** tool-use / function-calling via Ollama (known litellm quirk: `tool_choice` / `functions` hang)
- **Out of scope:** changing `get_response_from_llm` signature — Ollama passes through the default branch unchanged

## Context & Research

### Relevant Code and Patterns

- `agent/llm.py` — model constants use `"provider/model-name"` format; `get_response_from_llm` default branch applies to Ollama (temperature ✓, max_tokens → `num_predict` ✓, litellm.drop_params=True handles unsupported params)
- `tests/test_oauth_integration.py` — fixture pattern (`scope="module"`, three-mode: pre-configured / already-running / start-fresh), `pytestmark` module-level skip, `TestXxxHealth` + `TestLiveXxxCall` class grouping
- `tests/conftest.py` — `proxy_preconfigured()` helper pattern to replicate for Ollama
- `Dockerfile` line 144 — existing `ANTHROPIC_AUTH_MODE= python -m pytest ... --ignore=tests/test_oauth_integration.py` to extend with `--ignore=tests/test_ollama_integration.py`

### Key litellm Facts (1.74.9)

- **Use `ollama_chat/` prefix**, not `ollama/`**: routes to `/api/chat` which supports message history; `ollama/` uses `/api/generate` and mangles multi-turn prompts for instruct models
- **`OLLAMA_API_BASE`** is the canonical env var; fallback chain is kwarg → env var → `http://localhost:11434`
- **Health check:** `GET {base}/api/tags` → HTTP 200 (also lists pulled models; useful for skipping tests when model not pulled)
- **`max_tokens` maps to `num_predict`**; Ollama's default `num_predict=128` is extremely short — always pass explicit `max_tokens`
- **`litellm.drop_params = True`** already set globally — Ollama-unsupported OpenAI params are silently dropped
- **No API key needed** — litellm returns `None` for Ollama's API key unconditionally

### Qwen2.5-Coder 7B Tags

| Tag | Size | Notes |
|-----|------|-------|
| `qwen2.5-coder:7b-instruct-q4_K_M` | ~4.7 GB | **Default — best quality/size for M4 16 GB** |
| `qwen2.5-coder:7b-instruct-q5_K_M` | ~5.0 GB | Better quality, still fits M4 16 GB |
| `qwen2.5-coder:7b-instruct-q8_0` | ~7.7 GB | Near fp16, requires M4 Pro/Max |
| `qwen2.5-coder:7b` | ~4.7 GB | Alias for 7b-instruct (Ollama's auto-quant) |

Metal acceleration is automatic on macOS — no `num_gpu` flag needed.

## Key Technical Decisions

- **`ollama_chat/` over `ollama/`:** The codebase passes a full `messages` list; `ollama_chat/` is the only prefix that preserves multi-turn history correctly
- **Env var `OLLAMA_MODEL` drives the constant:** `agent/llm.py` reads `OLLAMA_MODEL` at import time with a default of `ollama_chat/qwen2.5-coder:7b-instruct-q4_K_M`; this lets users switch quantization without code changes
- **`OLLAMA_API_BASE` is the only config needed for Docker:** Since the container runs with `--network=host`, `http://localhost:11434` reaches the host Ollama — no special wiring beyond ensuring the env var is either set or left at default
- **No changes to `get_response_from_llm`:** The Ollama model string passes through the default branch cleanly; litellm handles the provider routing
- **Test skip uses reachability probe, not env var check:** Unlike OAuth (which requires explicit opt-in via `ANTHROPIC_AUTH_MODE`), Ollama availability is purely structural — if the endpoint is up, run the tests; if not, skip. This avoids requiring users to set extra env vars to opt in
- **Model-pulled check in fixture:** The `ollama_running` fixture should verify the specific model is present via `/api/tags` and skip with a helpful message if the model isn't pulled yet

## Open Questions

### Resolved During Planning

- **`ollama/` vs `ollama_chat/` prefix:** Resolved — use `ollama_chat/`; confirmed by litellm source that `ollama/` uses `/api/generate` which serializes the message list to a flat prompt string
- **Do we need a new dependency?** No — litellm 1.74.9 already ships the Ollama provider
- **Does `get_response_from_llm` need changes?** No — default branch handles Ollama correctly; `drop_params=True` handles any unsupported fields

### Deferred to Implementation

- **Exact model-pulled detection in fixture:** Check `/api/tags` JSON for model name presence; exact JSON structure is a runtime concern
- **Whether `ollama_chat/` model string with colons (`:`) causes any litellm routing issues:** Unlikely given how litellm strips the prefix, but verify during test execution

## High-Level Technical Design

> *This illustrates the intended approach and is directional guidance for review, not implementation specification. The implementing agent should treat it as context, not code to reproduce.*

```
.env
  OLLAMA_MODEL=ollama_chat/qwen2.5-coder:7b-instruct-q4_K_M  ← user-configurable
  OLLAMA_API_BASE=http://localhost:11434                        ← default, override for remote

agent/llm.py (import time)
  load_dotenv()
  OLLAMA_MODEL = os.getenv("OLLAMA_MODEL", "ollama_chat/qwen2.5-coder:7b-instruct-q4_K_M")
  ↓
  get_response_from_llm(msg, model=OLLAMA_MODEL)
    └─ litellm.completion(model="ollama_chat/qwen2.5-coder:7b-...", ...)
         └─ reads OLLAMA_API_BASE → http://localhost:11434/api/chat
              └─ Ollama (local process, Metal-accelerated)

tests/conftest.py
  ollama_preconfigured() → bool
    GET {OLLAMA_API_BASE}/api/tags → 200?

tests/test_ollama_integration.py
  pytestmark: skip if not ollama_preconfigured()
  ollama_running fixture: verify model tag present in /api/tags
  TestOllamaHealth: endpoint reachable, model pulled
  TestLiveOllamaCall: basic response, coherence, history, max_tokens, temperature=0

Dockerfile test step
  RUN ANTHROPIC_AUTH_MODE= python -m pytest tests/ \
      --ignore=tests/test_oauth_integration.py \
      --ignore=tests/test_ollama_integration.py \   ← add this
      -q --tb=short
```

## Implementation Units

- [ ] **Unit 1: Add OLLAMA_MODEL constant and env var wiring to `agent/llm.py`**

  **Goal:** Export a configurable `OLLAMA_MODEL` constant that follows the existing provider constant pattern and is usable as a drop-in `model=` argument.

  **Requirements:** R1, R2, R3, R4

  **Dependencies:** None

  **Files:**
  - Modify: `agent/llm.py`
  - Modify: `.env` (add commented example lines for Ollama vars)

  **Approach:**
  - After `load_dotenv()`, read `OLLAMA_MODEL` from env with default `"ollama_chat/qwen2.5-coder:7b-instruct-q4_K_M"`
  - Export as module-level constant `OLLAMA_MODEL` alongside the existing `CLAUDE_MODEL`, `OPENAI_MODEL` constants
  - `OLLAMA_API_BASE` requires no explicit handling in `llm.py` — litellm reads it directly from `os.environ`; just document it in `.env`
  - No changes to `get_response_from_llm` body

  **Patterns to follow:**
  - Existing constant block in `agent/llm.py`: `CLAUDE_MODEL = "anthropic/..."`, `OPENAI_MODEL = "openai/..."` — follow exactly

  **Test scenarios:**
  - `OLLAMA_MODEL` env var unset → constant resolves to `q4_K_M` default
  - `OLLAMA_MODEL=ollama_chat/qwen2.5-coder:7b` → constant resolves to that value
  - Constant is importable: `from agent.llm import OLLAMA_MODEL`

  **Verification:**
  - `OLLAMA_MODEL` is exported and importable
  - Default value is `"ollama_chat/qwen2.5-coder:7b-instruct-q4_K_M"`
  - Setting `OLLAMA_MODEL` env var before import overrides the default

---

- [ ] **Unit 2: Add `ollama_preconfigured()` helper to `tests/conftest.py`**

  **Goal:** Give all test modules a shared, importable probe that checks whether a reachable Ollama instance is available.

  **Requirements:** R5, R6

  **Dependencies:** Unit 1 (need the default base URL logic to match)

  **Files:**
  - Modify: `tests/conftest.py`

  **Approach:**
  - Add `ollama_preconfigured() -> bool` function (not a fixture — plain callable, same pattern as `proxy_preconfigured`)
  - Read `OLLAMA_API_BASE` from env, default `http://localhost:11434`
  - Probe `GET {base}/api/tags` with a short timeout (2s); return `True` on HTTP 200, `False` on any error
  - Use `httpx` (already in requirements)

  **Patterns to follow:**
  - `proxy_preconfigured()` in `tests/conftest.py` — same structure, same return type

  **Test scenarios:**
  - Ollama not running → `ollama_preconfigured()` returns `False`
  - Ollama running → returns `True`
  - `OLLAMA_API_BASE` overrides the probed URL

  **Verification:**
  - `from conftest import ollama_preconfigured` works in test files
  - Returns `False` when no Ollama process is present (no exception raised)

---

- [ ] **Unit 3: Write `tests/test_ollama_integration.py`**

  **Goal:** Provide integration tests for the Ollama provider that skip automatically when Ollama is unavailable and work identically in local and Docker environments.

  **Requirements:** R5, R6

  **Dependencies:** Units 1 and 2

  **Files:**
  - Create: `tests/test_ollama_integration.py`

  **Approach:**
  - Module-level `pytestmark = pytest.mark.skipif(not ollama_preconfigured(), ...)` — skip entire module if Ollama unreachable
  - `ollama_running` fixture (`scope="module"`): verify target model is present in `/api/tags` JSON (skip with actionable message if not pulled: `ollama pull qwen2.5-coder:7b-instruct-q4_K_M`)
  - `TestOllamaHealth` class: endpoint reachable (200 from `/api/tags`), model tag present in the tags list
  - `TestLiveOllamaCall` class: basic non-empty response, coherence ("2+2" → "4"), multi-turn history, max_tokens respected, temperature=0 determinism
  - All live call tests import `get_response_from_llm, OLLAMA_MODEL` from `agent.llm` inside test methods (not top-level) to avoid module-level ccproxy side effects

  **Patterns to follow:**
  - `tests/test_oauth_integration.py` — class structure, fixture pattern, `pytestmark`, import-inside-method convention
  - `TestLiveClaudeCall` test scenarios — replicate all five call tests verbatim (same prompts, same assertions)

  **Test scenarios:**
  - `TestOllamaHealth.test_ollama_endpoint_reachable` — `/api/tags` returns 200
  - `TestOllamaHealth.test_target_model_pulled` — model tag present in tags response
  - `TestLiveOllamaCall.test_basic_response` — non-empty string returned
  - `TestLiveOllamaCall.test_response_is_coherent` — "2 + 2" → response contains "4"
  - `TestLiveOllamaCall.test_message_history_preserved` — multi-turn: set name, recall name
  - `TestLiveOllamaCall.test_max_tokens_respected` — `max_tokens=50` → response < 500 chars
  - `TestLiveOllamaCall.test_temperature_zero_deterministic` — two calls at `temperature=0.0` return same answer

  **Verification:**
  - All tests pass when Ollama is running with the target model pulled
  - All tests show as `SKIPPED` (not `ERROR`) when Ollama is not running
  - Tests are skipped during Docker image build (not collected)
  - Tests pass when run inside a container with `--network=host` against host Ollama

---

- [ ] **Unit 4: Update Dockerfile test step**

  **Goal:** Prevent the Ollama integration tests from being collected during the Docker image build (Ollama is not running at build time).

  **Requirements:** R7

  **Dependencies:** Unit 3

  **Files:**
  - Modify: `Dockerfile`

  **Approach:**
  - Append `--ignore=tests/test_ollama_integration.py` to the existing `RUN ... python -m pytest` step
  - No other Dockerfile changes — Ollama runs on the host, containers access it via `--network=host` at runtime

  **Patterns to follow:**
  - Existing `--ignore=tests/test_oauth_integration.py` flag in the same `RUN` step

  **Test scenarios:**
  - Docker build completes without Ollama present
  - Unit tests (llm, domain_utils, common, llm_withtools) still run during build

  **Verification:**
  - `docker build` succeeds without Ollama available
  - The pytest output during build shows the four unit test files passing; Ollama integration tests not mentioned

---

- [ ] **Unit 5: Update README with Ollama setup instructions**

  **Goal:** Give users a clear path from zero to running Ollama-backed experiments.

  **Requirements:** R8

  **Dependencies:** Units 1–3

  **Files:**
  - Modify: `README.md`

  **Approach:**
  - Add an "Option C: Ollama (Local Model)" section alongside the existing API key and OAuth sections
  - Cover: install Ollama (`brew install ollama` / Linux apt), pull the default model, set env var to use a different quantization, run the integration tests
  - Note: for Docker, start Ollama on the host before running the container; no container-side install needed
  - Include a one-liner for changing the model: `OLLAMA_MODEL=ollama_chat/qwen2.5-coder:7b-instruct-q5_K_M`

  **Patterns to follow:**
  - Existing "Option B: OAuth" section structure in README

  **Test scenarios:**
  - (Documentation — no automated tests)

  **Verification:**
  - README renders correctly on GitHub
  - Pull command, env var override, and test invocation are all present and accurate

## System-Wide Impact

- **Interaction graph:** `agent/llm.py` module-level `load_dotenv()` already runs; `OLLAMA_MODEL` constant is evaluated at import time, same as all other model constants — no new side effects
- **Error propagation:** If `OLLAMA_API_BASE` points at an unreachable host and a live call is made, litellm raises `httpx.ConnectError`; `get_response_from_llm` has `@backoff` retry — this will retry up to the backoff limit before surfacing to the caller, which is the existing behavior for all providers
- **State lifecycle risks:** None — Ollama is stateless per-request; no process management (unlike ccproxy)
- **API surface parity:** `OLLAMA_MODEL` constant is available for import anywhere that `CLAUDE_MODEL` is used today — experiment harnesses can adopt it without `get_response_from_llm` changes
- **Integration coverage:** Docker `--network=host` is already used by the polyglot harness; Ollama connectivity follows the same path

## Risks & Dependencies

- **Ollama not installed on the test machine:** Both pytest and Docker runtime gracefully skip; not a build risk
- **Model not pulled (`ollama pull` not run):** The `ollama_running` fixture detects this via `/api/tags` and skips with an actionable message — tests won't hang or error misleadingly
- **litellm version drift:** `requirements.txt` pins `1.74.9` but venv has `1.82.6`; both confirmed to support `ollama_chat/`. No risk here, but the drift is worth resolving separately
- **M4 Metal inference latency:** `temperature=0` determinism test may exhibit non-determinism at very low `max_tokens` due to Metal threading; if flaky, the test should be loosened to tolerate minor variation (e.g., strip/lowercase comparison)
- **Ollama model tag format with colons:** litellm strips the `ollama_chat/` prefix and passes the remainder verbatim to the Ollama API; confirm during implementation that `qwen2.5-coder:7b-instruct-q4_K_M` routes correctly

## Documentation / Operational Notes

- Users on M4 with 16 GB unified memory: `q4_K_M` (default) or `q5_K_M` are both safe
- Users on M4 Pro/Max (36 GB+): `q8_0` gives near-fp16 quality
- Ollama runs as a background service (`ollama serve`); macOS users with the Ollama.app installed get this automatically
- To run integration tests against a remote Ollama instance: `OLLAMA_API_BASE=http://192.168.x.x:11434 pytest tests/test_ollama_integration.py`

## Sources & References

- litellm Ollama provider source: `venv_nat/lib/python3.12/site-packages/litellm/llms/ollama/`
- litellm OLLAMA_API_BASE resolution: `ollama/common_utils.py` line 68
- `ollama_chat/` routes to `/api/chat`: `ollama/chat/transformation.py`
- Qwen2.5-Coder model tags: https://ollama.com/library/qwen2.5-coder/tags
- Related code: `agent/llm.py`, `tests/test_oauth_integration.py`, `tests/conftest.py`
