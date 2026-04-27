#!/usr/bin/env bash
# ===========================================================================
# build_deps.sh — layered source-build pipeline for SGLang on free-threaded
# CPython 3.14t.
#
# A clean `uv pip install sglang` cannot succeed on a fresh cp314t
# environment today: a large fraction of our compiled deps publish no
# cp314t wheels yet. This script bridges that gap by building everything
# from source in dependency order. It is also what `docker/Dockerfile.nogil`
# bakes into the `sglang/nogil:cu124-py314t` image and what
# `.github/workflows/nogil-build.yml` runs in CI.
#
# Stages (failing one prints the upstream tracking link and exits):
#
#   A  toolchain check            (python3.14t / cmake / rustc / nvcc / gcc)
#   B  prebuilt cp314t wheels     (numpy, scipy, build backends, ...)
#   C  PyTorch nightly stack      (torch, torchvision, torchaudio, torchao)
#   D  low-level C / Rust extensions (independent, can run in parallel)
#   E  CUDA kernels               (ordered: cuda-python -> flashinfer -> ...)
#   F  SGLang-owned components    (sgl-kernel, sgl-router)
#   G  install sglang itself      (--no-deps, --no-build-isolation)
#
# Usage:
#   bash scripts/nogil/build_deps.sh                 # all stages
#   bash scripts/nogil/build_deps.sh A B C           # subset
#   STAGES="E F" bash scripts/nogil/build_deps.sh    # equivalent
#
# Environment knobs:
#   PYTHON              python interpreter to install into (default: python3.14t)
#   PYTORCH_INDEX_URL   torch nightly index (default: cu124 nightly)
#   PYTORCH_NIGHTLY_DATE optional pin like "20260115"; not enforced when empty
#   TORCH_CUDA_ARCH_LIST CUDA arches to compile for (default: 8.0;9.0)
#   MAX_JOBS            parallel compile jobs (default: nproc)
#   SGL_REPO_ROOT       path to the sglang clone (default: auto-detected)
# ===========================================================================

set -euo pipefail

# ---------------------------------------------------------------------------
# Locate the repo root so we can find requirements files and ./sgl-kernel.
# ---------------------------------------------------------------------------
SCRIPT_DIR="$(cd -- "$(dirname -- "${BASH_SOURCE[0]}")" &>/dev/null && pwd)"
# Default: scripts/nogil/ -> scripts/ -> repo root.
SGL_REPO_ROOT="${SGL_REPO_ROOT:-$(cd -- "${SCRIPT_DIR}/../.." &>/dev/null && pwd)}"
NOGIL_REQS="${SGL_REPO_ROOT}/python/requirements/nogil"
PATCHES_DIR="${NOGIL_REQS}/patches"

PYTHON="${PYTHON:-python3.14t}"
PYTORCH_INDEX_URL="${PYTORCH_INDEX_URL:-https://download.pytorch.org/whl/nightly/cu124}"
PYTORCH_NIGHTLY_DATE="${PYTORCH_NIGHTLY_DATE:-}"
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;9.0}"
export MAX_JOBS="${MAX_JOBS:-$(nproc)}"

# ---------------------------------------------------------------------------
# Logging helpers.
# ---------------------------------------------------------------------------
RED=$'\033[0;31m'; GREEN=$'\033[0;32m'; YELLOW=$'\033[0;33m'
BLUE=$'\033[0;34m'; BOLD=$'\033[1m'; NC=$'\033[0m'

log()   { printf '%s[nogil]%s %s\n' "${BLUE}" "${NC}" "$*"; }
ok()    { printf '%s[ ok ]%s %s\n' "${GREEN}" "${NC}" "$*"; }
warn()  { printf '%s[warn]%s %s\n' "${YELLOW}" "${NC}" "$*"; }
fail()  { printf '%s[fail]%s %s\n' "${RED}" "${NC}" "$*" >&2; exit 1; }
stage() { printf '\n%s%s===== Stage %s =====%s\n' "${BOLD}" "${BLUE}" "$*" "${NC}"; }

# pip wrapper that always uses the chosen interpreter, never the active one.
pip_install() { "${PYTHON}" -m pip install "$@"; }

# Wrap any failing command with a clear "see this upstream link" message.
require() {
  local link="$1"; shift
  if ! "$@"; then
    warn "command failed: $*"
    warn "tracking: ${link}"
    fail "stage aborted"
  fi
}

# ---------------------------------------------------------------------------
# Stage A — toolchain check.
# ---------------------------------------------------------------------------
stage_A() {
  stage "A — toolchain check"

  # python3.14t
  if ! command -v "${PYTHON}" >/dev/null 2>&1; then
    fail "${PYTHON} not found. Install with:
    pyenv install 3.14.0t && pyenv shell 3.14.0t
  or build CPython with: ./configure --disable-gil --enable-optimizations"
  fi
  if ! "${PYTHON}" -c 'import sysconfig,sys; sys.exit(0 if sysconfig.get_config_var("Py_GIL_DISABLED") else 1)'; then
    fail "${PYTHON} is not a free-threaded build (Py_GIL_DISABLED is unset)."
  fi
  ok "$(${PYTHON} -VV | head -1) — Py_GIL_DISABLED=1"

  # cmake >= 3.27, ninja, rustc >= 1.78, nvcc >= 12.4, gcc >= 11
  local missing=()
  command -v cmake  >/dev/null 2>&1 || missing+=("cmake (apt install cmake; or pip install 'cmake>=3.27')")
  command -v ninja  >/dev/null 2>&1 || missing+=("ninja (apt install ninja-build)")
  command -v rustc  >/dev/null 2>&1 || missing+=("rustc (curl https://sh.rustup.rs -sSf | sh)")
  command -v nvcc   >/dev/null 2>&1 || missing+=("nvcc  (install CUDA toolkit >= 12.4)")
  command -v gcc    >/dev/null 2>&1 || missing+=("gcc   (apt install gcc-11 g++-11)")
  if (( ${#missing[@]} )); then
    printf '%s[fail]%s missing toolchain components:\n' "${RED}" "${NC}" >&2
    printf '         - %s\n' "${missing[@]}" >&2
    exit 1
  fi
  ok "cmake $(cmake --version | head -1)"
  ok "ninja $(ninja --version)"
  ok "rustc $(rustc --version)"
  ok "nvcc  $(nvcc --version | tail -2 | head -1 | xargs)"
  ok "gcc   $(gcc --version | head -1)"

  # Ensure pip is fresh inside this interpreter — required for PEP 658 / metadata 2.4.
  pip_install --upgrade pip
}

# ---------------------------------------------------------------------------
# Stage B — install everything that already has a cp314t wheel.
# ---------------------------------------------------------------------------
stage_B() {
  stage "B — prebuilt cp314t wheels"
  local req="${NOGIL_REQS}/wheels-available.txt"
  [[ -f "${req}" ]] || fail "missing ${req}"
  pip_install -r "${req}"
  ok "wheels-available.txt installed"
}

# ---------------------------------------------------------------------------
# Stage C — PyTorch nightly stack.
# ---------------------------------------------------------------------------
stage_C() {
  stage "C — PyTorch nightly (cp314t)"
  local pkgs=(torch torchvision torchaudio torchao)
  if [[ -n "${PYTORCH_NIGHTLY_DATE}" ]]; then
    # Map nightly date to the +YYYYMMDD local-version segment that the
    # download.pytorch.org/whl/nightly index advertises.
    pkgs=(
      "torch==*+${PYTORCH_NIGHTLY_DATE}*"
      "torchvision==*+${PYTORCH_NIGHTLY_DATE}*"
      "torchaudio==*+${PYTORCH_NIGHTLY_DATE}*"
      torchao
    )
  fi
  pip_install --pre "${pkgs[@]}" --index-url "${PYTORCH_INDEX_URL}" \
    || fail "PyTorch nightly install failed. See https://github.com/pytorch/pytorch/issues/130249"
  "${PYTHON}" -c 'import torch; print("[ ok ] torch", torch.__version__, "cuda", torch.version.cuda)'
}

# ---------------------------------------------------------------------------
# Helpers for parsing build-from-source.txt records.
# ---------------------------------------------------------------------------
# Reads `build-from-source.txt`, prints `<stage>|<name>|<patch>|<url>|<tracking>`
# lines for entries whose `stage:` matches $1.
parse_records() {
  local want_stage="$1"
  local req="${NOGIL_REQS}/build-from-source.txt"
  awk -v want="${want_stage}" '
    BEGIN { name=""; stg=""; patch=""; url=""; track="-" }
    /^---[[:space:]]*$/ {
      if (name != "" && stg == want) print stg "|" name "|" patch "|" url "|" track
      name=""; stg=""; patch=""; url=""; track="-"; next
    }
    /^[[:space:]]*#/ || NF==0 { next }
    /^name[[:space:]]*:/    { sub(/^name[[:space:]]*:[[:space:]]*/, ""); name=$0; next }
    /^stage[[:space:]]*:/   { sub(/^stage[[:space:]]*:[[:space:]]*/, ""); stg=$0; next }
    /^patch[[:space:]]*:/   { sub(/^patch[[:space:]]*:[[:space:]]*/, ""); patch=$0; next }
    /^url[[:space:]]*:/     { sub(/^url[[:space:]]*:[[:space:]]*/, ""); url=$0; next }
    /^tracking[[:space:]]*:/{ sub(/^tracking[[:space:]]*:[[:space:]]*/, ""); track=$0; next }
    END {
      if (name != "" && stg == want) print stg "|" name "|" patch "|" url "|" track
    }
  ' "${req}"
}

# Install one PyPI package from source. If a patch is named, ensure it
# exists on disk (build_deps.sh does not auto-clone; the patch consumer
# would have to be a separate `clone-and-patch` helper, which is overkill
# for the current set since `pip install --no-binary` covers all sdists
# we need today).
install_one_source() {
  local name="$1" patch="$2" url="$3" tracking="$4"
  log "building ${name} from source (url=${url})"
  if [[ -n "${patch}" ]]; then
    [[ -f "${PATCHES_DIR}/${patch}" ]] \
      || fail "${name}: referenced patch ${patch} not found in ${PATCHES_DIR}"
    warn "${name}: patch ${patch} must be applied to the sdist by hand or via a clone+apply wrapper"
    warn "${name}: tracking ${tracking}"
  fi

  # `name` is one of `pkg`, `pkg==X`, `pkg>=X`, none of which contain spaces,
  # so this is a defensive guard for future entries with extra trailing text.
  local pkg="${name%% *}"
  require "${tracking:-https://github.com/sgl-project/sglang}" \
    pip_install --no-build-isolation --no-binary "${pkg}" "${name}"
}

# ---------------------------------------------------------------------------
# Stage D — low-level C / Rust extensions.
# ---------------------------------------------------------------------------
stage_D() {
  stage "D — low-level C / Rust extensions"
  while IFS='|' read -r _ name patch url tracking; do
    [[ -z "${name}" ]] && continue
    install_one_source "${name}" "${patch}" "${url}" "${tracking}"
  done < <(parse_records D)
  ok "Stage D complete"
}

# ---------------------------------------------------------------------------
# Stage E — CUDA kernels (in declared order).
# ---------------------------------------------------------------------------
stage_E() {
  stage "E — CUDA kernels"
  while IFS='|' read -r _ name patch url tracking; do
    [[ -z "${name}" ]] && continue
    install_one_source "${name}" "${patch}" "${url}" "${tracking}"
  done < <(parse_records E)
  ok "Stage E complete"
}

# ---------------------------------------------------------------------------
# Stage F — SGLang-owned compiled components.
# ---------------------------------------------------------------------------
stage_F() {
  stage "F — SGLang-owned components"

  log "building sgl-kernel with SGL_KERNEL_NOGIL=ON"
  (
    cd "${SGL_REPO_ROOT}/sgl-kernel"
    # Drop the `cp310` stable-ABI restriction for free-threaded builds.
    # See sgl-kernel/CMakeLists.txt where the SGL_KERNEL_NOGIL option is
    # consumed. SKBUILD_WHEEL_PY_API="" tells scikit-build-core to ignore
    # `wheel.py-api = "cp310"` from pyproject.toml so we get a real
    # cp314t-tagged wheel instead of an abi3 one (which is invalid for
    # free-threaded builds).
    SKBUILD_WHEEL_PY_API="" \
    SKBUILD_CMAKE_DEFINE="SGL_KERNEL_NOGIL=ON" \
      pip_install --no-build-isolation .
  )
  ok "sgl-kernel installed"

  if [[ -d "${SGL_REPO_ROOT}/sgl-model-gateway" ]]; then
    log "building sgl-router (PyO3, gil_used=false)"
    (
      cd "${SGL_REPO_ROOT}/sgl-model-gateway"
      pip_install --no-build-isolation .
    )
    ok "sgl-router installed"
  else
    warn "sgl-model-gateway/ not present in this checkout — skipping sgl-router"
  fi
}

# ---------------------------------------------------------------------------
# Stage G — install sglang itself.
# ---------------------------------------------------------------------------
stage_G() {
  stage "G — install sglang"
  # --no-deps because Stages B–F already resolved every transitive dep; if
  # we let pip resolve again it will go to PyPI for cp314t wheels that do
  # not exist and the install will explode.
  pip_install --no-build-isolation --no-deps -e "${SGL_REPO_ROOT}/python"
  ok "sglang installed"
  "${PYTHON}" -c 'import sys; assert not sys._is_gil_enabled(), "GIL still enabled"; \
import sglang; print("[ ok ] sglang", getattr(sglang, "__version__", "dev"), "on", sys.version)'
}

# ---------------------------------------------------------------------------
# Stage dispatch.
# ---------------------------------------------------------------------------
ALL_STAGES=(A B C D E F G)
SELECTED=("${@:-${STAGES:-${ALL_STAGES[@]}}}")

for s in "${SELECTED[@]}"; do
  case "${s}" in
    A) stage_A ;;
    B) stage_B ;;
    C) stage_C ;;
    D) stage_D ;;
    E) stage_E ;;
    F) stage_F ;;
    G) stage_G ;;
    *) fail "unknown stage '${s}' (valid: ${ALL_STAGES[*]})" ;;
  esac
done

ok "build_deps.sh: stages ${SELECTED[*]} completed"
