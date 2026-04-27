# SGLang dependency manifest for free-threaded CPython 3.14t

This directory is the machine-readable source of truth for what SGLang needs
in a free-threaded (`cp314t`, `Py_GIL_DISABLED`) Python environment. It is
consumed by:

* `scripts/nogil/build_deps.sh` — the layered source-build script that turns
  a clean Linux + CUDA host into a working `python3.14t` venv with SGLang
  installed.
* `docker/Dockerfile.nogil` — wraps the script in a reproducible image.
* `.github/workflows/nogil-build.yml` — CI guard that re-runs the pipeline
  whenever any file under this directory, `scripts/nogil/`, `sgl-kernel/`,
  or `docker/Dockerfile.nogil` changes.

The split into two requirement files is **deliberate**: a clean
`uv pip install sglang` cannot succeed on a fresh cp314t environment today
because a large fraction of our compiled deps publish no cp314t wheels. The
two files draw the line between "PyPI has it" and "you must build it".

## Files

### `wheels-available.txt`

Packages that publish cp314t wheels on PyPI (or one of the well-known mirrors
like the PyTorch nightly index). Installable directly with
`uv pip install -r wheels-available.txt`. Versions are pinned to the **lowest
known-good** release that ships a `cp314t` wheel — bumping these is a
supply-chain change and must be done explicitly.

### `build-from-source.txt`

Packages that **must** be built from source on cp314t. Each entry is annotated
with:

* upstream repository / sdist URL
* required commit / tag (or "latest" with the rationale)
* whether a patch from `patches/` must be applied first
* native build dependencies (CMake / Ninja / Rust / `nvcc` ≥ X / libsndfile / …)

This file is **not** consumed by `pip` directly — `build_deps.sh` parses the
annotations and runs `pip install --no-build-isolation` in the right order.

### `patches/`

Out-of-tree patches kept here only until upstream merges the corresponding
free-threading fix. Each patch's first line points at the upstream PR or
issue, so it is obvious when the patch can be deleted.

## Validation matrix

| Tier | Goal | Tested by |
|------|------|-----------|
| 0 | `wheels-available.txt` resolves | `uv pip install --dry-run -r wheels-available.txt` |
| 1 | `build_deps.sh` Stages A–G complete | `nogil-build.yml` |
| 2 | Server smoke test on Qwen2.5-0.5B | `nogil-build.yml` |
| 3 | `sys._is_gil_enabled()` is `False` in the server process | `nogil-build.yml` |

Anything beyond tier 3 (full unit tests, perf benchmarks) belongs to Phase 5
and Phase 6 of the RFC, not to this directory.
