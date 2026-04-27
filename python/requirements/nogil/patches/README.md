# Out-of-tree patches for free-threaded (cp314t) builds

Each `*.patch` file in this directory is a temporary fix applied to a
third-party source tree before it is built against `cp314t`. The patches
exist only because upstream has not yet merged the equivalent change.

## Conventions

* Filename: `<package-name>-py314t.patch`.
* The first line of every patch **MUST** be a `# upstream:` comment that
  links to the corresponding upstream PR or issue. When that link is closed
  / merged, the patch is obsolete and must be removed (and the corresponding
  pin in `../build-from-source.txt` bumped past the fix).
* Patches are applied with `git apply --3way` from the root of the cloned
  source tree. They must therefore be generated with `git diff --no-prefix`
  or with `-p1` paths that match a fresh `git clone`.
* No patch may add a runtime dependency. They are strictly for fixing
  build-time / ABI issues (e.g. dropping `Py_LIMITED_API`, switching off
  `gil_scoped_release` no-ops, swapping `PyDict_SetItem` for thread-safe
  variants).

## Currently expected patches

The matching entries in `../build-from-source.txt` reference these
filenames; `build_deps.sh` will fail loudly if a referenced patch is missing.

| Filename | Tracks |
|----------|--------|
| `outlines-core-py314t.patch` | https://github.com/dottxt-ai/outlines-core/issues/248 |
| `quack-kernels-py314t.patch` | https://github.com/Dao-AILab/quack (no issue yet) |

When upstream merges either fix, delete the `.patch`, drop the `patch:`
line from `../build-from-source.txt`, and bump the pinned version.
