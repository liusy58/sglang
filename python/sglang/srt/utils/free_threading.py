"""Helpers for running SGLang on a free-threaded (PEP 703 / nogil) CPython build.

This module is intentionally tiny and has *no* third-party imports so it can be
imported very early during process start-up (before fastapi / torch / uvloop).

Two pieces of state are exposed:

* :func:`is_free_threaded`  - True when the running interpreter is the
  ``cpXYt`` free-threaded build (e.g. ``python3.14t``). Detected via
  ``sysconfig`` so the result is correct even when the GIL has been
  re-enabled at runtime by an extension that did not opt in to nogil.

* :func:`is_nogil_mode`     - True when the user has explicitly opted in
  to the experimental nogil bring-up profile by exporting
  ``SGLANG_NOGIL=1``. This is used to *gate* fall-back paths (e.g. skip
  ``uvloop`` even on a regular GIL build for testing) so the regular
  install behaves identically when the env var is unset.

The :func:`set_uvloop_policy_if_available` helper centralises the uvloop
opt-in logic that used to live verbatim in three different modules
(``entrypoints/engine.py``, ``entrypoints/http_server.py`` and
``managers/tokenizer_manager.py``). uvloop currently has no support for
free-threaded Python, so on a cp3XYt interpreter (or when
``SGLANG_NOGIL=1`` is set) we silently fall back to the stdlib asyncio
event loop policy instead of crashing at import time.
"""

from __future__ import annotations

import asyncio
import logging
import os
import sysconfig

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Detection helpers
# ---------------------------------------------------------------------------

# Cache the sysconfig probe — it is constant for the lifetime of the process
# and is consulted on every server start-up.
_FREE_THREADED: bool = bool(sysconfig.get_config_var("Py_GIL_DISABLED"))


def is_free_threaded() -> bool:
    """Return ``True`` on a free-threaded (``cpXYt``) CPython build.

    This is independent of whether the GIL is currently enabled at runtime:
    extensions that did not declare ``Py_mod_gil = Py_MOD_GIL_NOT_USED`` will
    cause CPython to re-enable the GIL after import, but the interpreter
    binary is still a free-threaded build.
    """

    return _FREE_THREADED


def is_nogil_mode() -> bool:
    """Return ``True`` when SGLang's experimental nogil bring-up is requested.

    Two ways to opt in:
      * Run on a free-threaded interpreter (``python3.14t``); detected
        automatically.
      * Export ``SGLANG_NOGIL=1`` on a regular GIL build (useful for
        testing the optional-dependency fall-back paths without having to
        rebuild CPython).
    """

    if is_free_threaded():
        return True
    return os.environ.get("SGLANG_NOGIL", "").strip().lower() in ("1", "true", "yes", "on")


# ---------------------------------------------------------------------------
# uvloop fall-back
# ---------------------------------------------------------------------------

# Sentinel so we only log the "uvloop disabled" message once per process even
# if multiple subsystems call into this helper.
_uvloop_logged: bool = False


def set_uvloop_policy_if_available() -> bool:
    """Install uvloop's event loop policy when it is safe to do so.

    Returns ``True`` if uvloop was activated, ``False`` otherwise.

    uvloop is skipped in three cases:

    1. The interpreter is a free-threaded (nogil) build. uvloop has no
       support for PEP 703 yet and importing it on cp3XYt either fails
       outright or silently re-enables the GIL.
    2. ``SGLANG_NOGIL=1`` is set in the environment. This is the manual
       opt-in for testing the fall-back path on a regular GIL build.
    3. ``uvloop`` is not installed (e.g. the user followed the nogil
       :file:`pyproject.nogil.toml` profile which intentionally omits it).

    In all skip cases the stdlib asyncio default event loop policy is
    left in place, which is fully functional — only somewhat slower than
    uvloop.
    """

    global _uvloop_logged

    if is_nogil_mode():
        if not _uvloop_logged:
            logger.info(
                "uvloop is disabled: free-threaded Python / SGLANG_NOGIL=1 detected; "
                "falling back to the stdlib asyncio event loop policy."
            )
            _uvloop_logged = True
        return False

    try:
        import uvloop  # type: ignore[import-not-found]
    except ImportError:
        if not _uvloop_logged:
            logger.info(
                "uvloop is not installed; falling back to the stdlib asyncio "
                "event loop policy."
            )
            _uvloop_logged = True
        return False

    asyncio.set_event_loop_policy(uvloop.EventLoopPolicy())
    return True
