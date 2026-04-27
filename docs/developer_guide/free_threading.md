# Running SGLang on Free-Threaded (cp314t) Python

> ⚠️ **Experimental.** Free-threaded CPython 3.14t (PEP 703) is the first
> release where the no-GIL build is stable enough to attempt serving
> experiments. Most of the deep-learning ecosystem still has gaps: this
> guide describes a research bring-up — not a production deployment path.

`uv pip install sglang` cannot succeed on a clean cp314t environment
today. A large fraction of our compiled dependencies (`torch*`,
`sgl-kernel`, `flash-attn`, `flashinfer*`, `cuda-python`, `quack-kernels`,
`soundfile`, `uvloop`, `setproctitle`, `pybase64`, `tiktoken`,
`outlines-core`, `sentencepiece`, …) ship no cp314t wheels. SGLang
therefore ships a **reproducible source-build pipeline** instead of
relying on PyPI.

There are two ways to use it:

1. [**Quick start (Docker)**](#quick-start-docker) — pull the prebuilt
   image and run.
2. [**From source (Stage A → G)**](#from-source-stage-a--g) — run the
   layered build script directly on a host with a CUDA toolkit.

Whichever path you pick, the four-tier validation matrix in
`python/requirements/nogil/README.md` is what defines "done".

---

## Quick start (Docker)

```bash
docker pull sglang/nogil:cu124-py314t

# Smoke check — should print "nogil ok" and the SGLang version.
docker run --rm --gpus all sglang/nogil:cu124-py314t \
  python3.14t -c 'import sys; \
    assert not sys._is_gil_enabled(), "GIL still enabled"; \
    import sglang; print("nogil ok,", sglang.__version__)'

# Serve a small text-only model.
docker run --rm --gpus all -p 30000:30000 sglang/nogil:cu124-py314t \
  -m sglang.launch_server \
    --model Qwen/Qwen2.5-0.5B \
    --port 30000 \
    --disable-cuda-graph \
    --attention-backend triton
```

The image is rebuilt by `.github/workflows/nogil-build.yml` on every
change to `python/requirements/nogil/**`, `scripts/nogil/**`,
`sgl-kernel/**`, or `docker/Dockerfile.nogil`, so its contents are
always in sync with `main`.

If you need a different CUDA version or want to bake in a model:

```bash
docker build -f docker/Dockerfile.nogil \
  --build-arg CUDA_VERSION=12.4.1 \
  --build-arg CPYTHON_VERSION=3.14.0 \
  -t my-sglang-nogil .
```

---

## From source (Stage A → G)

The pipeline lives in `scripts/nogil/build_deps.sh`. Each stage is
self-contained and prints the upstream tracking link when it fails so you
can debug or skip the offending dependency in isolation.

### Stage A — toolchain check

```bash
# python3.14t (free-threaded)
pyenv install 3.14.0t || ./configure --disable-gil --enable-optimizations

# system tools
sudo apt-get install -y \
    build-essential gcc-11 g++-11 \
    cmake ninja-build pkg-config patchelf \
    libssl-dev libffi-dev zlib1g-dev libsndfile1-dev \
    libuv1-dev libzmq3-dev libprotobuf-dev protobuf-compiler

# rustc >= 1.78
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh

# nvcc >= 12.4 (CUDA toolkit)
# follow https://developer.nvidia.com/cuda-downloads
```

Rerun `bash scripts/nogil/build_deps.sh A` until it prints `[ ok ]` for
every component.

> **If it fails:** the script names the missing tool and the apt /
> rustup / nvidia install command. Do not skip Stage A — a missing
> compiler causes confusing failures three stages later.

### Stage B — prebuilt cp314t wheels

```bash
bash scripts/nogil/build_deps.sh B
```

Installs everything from `python/requirements/nogil/wheels-available.txt`
in a single resolver pass: `numpy`, `scipy`, `pyzmq`, `msgspec`,
`orjson`, `aiohttp`, `xgrammar`, `llguidance`, `xformers`, plus the build
backends (`scikit-build-core`, `pybind11`, `maturin`, `cython`, `cffi`)
that later stages need.

> **If it fails:** PyPI no longer publishes the pinned cp314t wheel for
> some package. Bump that line in `wheels-available.txt` and retry.

### Stage C — PyTorch nightly

```bash
bash scripts/nogil/build_deps.sh C
# or pin a date:
PYTORCH_NIGHTLY_DATE=20260115 bash scripts/nogil/build_deps.sh C
```

Pulls `torch` / `torchvision` / `torchaudio` / `torchao` from the
official cp314t nightly index. We do **not** build PyTorch from source
here (it would take hours); the lock file pins the nightly date instead.

> **If it fails:** the cp314t wheel for the requested date no longer
> exists. Check
> <https://download.pytorch.org/whl/nightly/cu124/torch/> and bump
> `PYTORCH_NIGHTLY_DATE` to a recent listed date.

### Stage D — low-level C / Rust extensions

```bash
bash scripts/nogil/build_deps.sh D
```

Builds `soundfile`, `pybase64`, `setproctitle`, `tiktoken`, `uvloop`,
`outlines-core`, `sentencepiece`, `compressed-tensors`,
`smg-grpc-servicer` from source. Entries that need an out-of-tree fix
reference a patch under `python/requirements/nogil/patches/`.

> **If it fails:** the script prints the upstream issue. If you need to
> unblock yourself, comment the failing record out of
> `build-from-source.txt` — every Stage D package is optional at runtime
> as long as the corresponding feature isn't exercised.

### Stage E — CUDA kernels

```bash
bash scripts/nogil/build_deps.sh E
```

Order matters here: `cuda-python` → `flashinfer-cubin` →
`flashinfer-python` → `flash-attn` → `quack-kernels` → `kernels`. All
build with `--no-build-isolation` so they reuse the cp314t torch from
Stage C instead of pulling a with-GIL one into a temp env.

> **If it fails:** verify `nvcc --version` is ≥ 12.4 and
> `TORCH_CUDA_ARCH_LIST` covers your GPU's compute capability
> (`8.0` for A100, `9.0` for H100).

### Stage F — SGLang-owned components

```bash
bash scripts/nogil/build_deps.sh F
```

Builds `sgl-kernel` with `SGL_KERNEL_NOGIL=ON`, which drops the cp310
stable-ABI restriction (`wheel.py-api = "cp310"` in
`sgl-kernel/pyproject.toml`) and emits a real `*.cpython-314t-*.so`. See
`sgl-kernel/CMakeLists.txt` for how the option flips the SABI clause off
at every `Python_add_library()` call site.

Also builds `sgl-router` (PyO3, already annotated `gil_used = false`) if
the `sgl-model-gateway/` directory is present.

> **If it fails:** the most common cause is mismatched `nvcc` ↔ `g++`
> versions. CUDA 12.4 wants `gcc-11` or `gcc-12`, not `gcc-13`.

### Stage G — install sglang

```bash
bash scripts/nogil/build_deps.sh G
```

`pip install -e python --no-deps --no-build-isolation`. The
`--no-deps` flag is critical: by this point Stages B–F have resolved
every transitive dependency; if we let pip resolve again it will go to
PyPI for cp314t wheels that don't exist and undo the whole install.

A successful Stage G ends with:

```
[ ok ] sglang 0.x.y on CPython 3.14.0 free-threading build
```

---

## Validation

The CI workflow `.github/workflows/nogil-build.yml` runs the same four
tiers documented in `python/requirements/nogil/README.md`:

| Tier | What |
|------|------|
| 0 | `wheels-available.txt` resolves on a stock Ubuntu runner |
| 1 | `build_deps.sh` Stages A–G complete inside the Docker image |
| 2 | Server starts, `/generate` returns sane output for Qwen2.5-0.5B |
| 3 | `sys._is_gil_enabled()` is `False` in the server process |

Anything past tier 3 (full unit tests, perf benchmarks) is Phase 5 / 6
of the RFC and intentionally **not** part of this workflow — it would
exceed the runner timeout.

---

## Relationship to earlier nogil work

The earlier nogil patches (`python/pyproject.nogil.toml`,
`python/sglang/srt/utils/free_threading.py`, the PyO3
`gil_used = false` annotation, the runtime `optional_import` guards) are
still in the tree but are **downstream** of this pipeline: they answer
"once everything is installed, can SGLang import and run?". This file —
together with `scripts/nogil/build_deps.sh`,
`docker/Dockerfile.nogil`, and `python/requirements/nogil/` — answers
the upstream question of "can it be installed at all?", which is what
RFC Phase 1 demands.

When all four validation tiers stay green for a full release cycle, we
can close Phase 1 and start Phase 2 (the proper `Py_GIL_DISABLED` audit
of `sgl-kernel`).
