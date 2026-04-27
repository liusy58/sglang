# Running SGLang on Free-Threaded (nogil) Python 3.14t

> ⚠️ **Experimental.** As of 2026-04, no major deep-learning stack ships an
> officially supported free-threaded build. This guide describes a research
> bring-up — not a production deployment path.

CPython 3.13 introduced an experimental free-threaded build (PEP 703) that
removes the Global Interpreter Lock (GIL). CPython 3.14t is the first release
where the build is stable enough to attempt serving experiments. This document
captures the SGLang-side knobs that exist today and the procedure for bringing
up a single-GPU text-only model (e.g. `Qwen3.5-A3B`) on `python3.14t`.

## What is in the repo

* **`python/pyproject.nogil.toml`** — a minimum-viable dependency manifest
  for the cp314t target. It removes every GIL-coupled / multimodal /
  quantization / disaggregation dependency and pins only what is strictly
  needed to run a text-only Qwen MoE model with the Triton attention
  backend.
* **`python/sglang/srt/utils/free_threading.py`** — runtime helpers:
  * `is_free_threaded()` — `True` on a `cpXYt` interpreter (detected via
    `sysconfig.get_config_var("Py_GIL_DISABLED")`).
  * `is_nogil_mode()` — `True` on a free-threaded interpreter **or** when
    `SGLANG_NOGIL=1` is exported. Use this to gate fall-back paths in your
    own code.
  * `set_uvloop_policy_if_available()` — installs uvloop's event loop
    policy when it is safe (uvloop has no PEP 703 support today, so on
    cp3XYt or under `SGLANG_NOGIL=1` we silently fall back to the stdlib
    asyncio policy). This is now the only place uvloop is referenced in
    SGLang's hot-path entry-points.
* **`rust/sglang-grpc/src/lib.rs`** — the PyO3 module is annotated with
  `#[pymodule(gil_used = false)]` so importing the gRPC extension on a
  free-threaded interpreter does not re-enable the GIL.

## Bring-up procedure

### 1. Toolchain

* CUDA 12.9 toolkit (`CUDA_HOME=/usr/local/cuda-12.9`)
* GCC/G++ ≥ 11, `ninja`, `cmake ≥ 3.26`, `patchelf`
* NVIDIA driver compatible with the cu130 PyTorch wheel
* `uv` (≥ 0.4)

### 2. Install free-threaded CPython and create a venv

```bash
uv python install 3.14t
uv venv --python 3.14t .venv-nogil
source .venv-nogil/bin/activate
python -VV          # must mention "free-threading build"
```

Upgrade the build chain:

```bash
uv pip install -U pip setuptools setuptools-scm wheel scikit-build-core \
    ninja cmake "pybind11>=2.13" "cython>=3.1" "maturin>=1.7"
```

Set compile flags:

```bash
export MAX_JOBS=$(nproc)
export TORCH_CUDA_ARCH_LIST="9.0"   # H100 example; 8.0 for A100
```

### 3. Use the nogil pyproject

```bash
cp python/pyproject.nogil.toml python/pyproject.toml.bak  # keep a backup
cp python/pyproject.nogil.toml python/pyproject.toml      # activate
```

(Do **not** commit this swap — it is per-venv only.)

### 4. Install dependencies (source builds where needed)

```bash
# numpy / scipy / pillow first — they ship cp314t wheels in 2.1+ / 1.14+ / 10.4+
uv pip install "numpy>=2.1" scipy pillow

# native extensions that almost always need a source build on cp314t
for pkg in pyzmq orjson msgspec pybase64 sentencepiece tiktoken \
           setproctitle watchfiles py-spy "cuda-python>=13.0"; do
  uv pip install --no-binary "$pkg" "$pkg"
done

# torch — try the official index first, fall back to a source build
uv pip install torch --index-url https://download.pytorch.org/whl/nightly/cu130 \
  || echo "Build torch from source (USE_CUDA=1 USE_DISTRIBUTED=1 ...)"

# triton attention backend
uv pip install --no-binary triton "triton>=3.2"
```

Then build SGLang's own kernel package:

```bash
cd sgl-kernel
pip install --no-build-isolation -e .
cd ..
```

And finally SGLang itself:

```bash
uv pip install --no-build-isolation -e python/
```

### 5. Launch the server

The actual launch line you would use to bring up Qwen3.5-A3B:

```bash
export CUDA_HOME=/usr/local/cuda-12.9
export SGLANG_TORCH_PROFILER_DIR=.
export SGLANG_NOGIL=1                     # enables fall-back paths

CUDA_VISIBLE_DEVICES=2 python -X gil=0 -m sglang.launch_server \
    --model-path /disk3/models/Qwen3.5-35B-A3B \
    --port 8000 --tp-size 1 --mem-fraction-static 0.9 \
    --attention-backend triton \
    --disable-cuda-graph \
    --disable-radix-cache \
    --grammar-backend none
```

In a second shell, confirm the GIL really is disabled:

```bash
python -X gil=0 -c "import sys; print('gil enabled?', sys._is_gil_enabled())"
```

If the answer is `True`, look at the SGLang stderr for the very first line
of the form

```
The global interpreter lock (GIL) has been enabled to load module '<name>'
```

and address that extension (upgrade version, add `Py_mod_gil = Py_MOD_GIL_NOT_USED`,
or rebuild against `pybind11 >= 2.13`).

## Optional dependencies

These packages are intentionally omitted from `pyproject.nogil.toml` because
either no cp314t wheel exists yet, or they are not needed for the text-only
Qwen MoE bring-up target. SGLang already guards their import sites with
`try/except` so the daemon starts cleanly when they are missing:

`flashinfer_python`, `flashinfer_cubin`, `flash-attn-4`, `quack-kernels`,
`torch_memory_saver`, `torchao`, `torchcodec`, `decord2`, `av`,
`apache-tvm-ffi`, `nvidia-cutlass-dsl`, `runai-model-streamer`,
`smg-grpc-servicer`, `kernels`, `mistral_common`, `soundfile`, `timm`,
`xgrammar`, `outlines`, `openai-harmony`, `anthropic`, `modelscope`, `gguf`,
`compressed-tensors`, `IPython`, `torchvision`, `torchaudio`.

Re-introduce them one by one once you have a clean baseline; each requires a
source build with the toolchain set up above.

## Plan B — Python 3.13t

If `cp314t` wheels are too sparse on your platform, the fastest fall-back is
`cp313t`, which has been stable for a year and where the PyTorch nightly
index already publishes free-threaded wheels. Replace `3.14t` with `3.13t`
in step 2 above; everything else (including `pyproject.nogil.toml` after
adjusting `requires-python`) is identical.
