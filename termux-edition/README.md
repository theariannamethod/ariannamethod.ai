# ariannamethod.ai — Termux Edition

**AML system-wide install on Android Termux (aarch64). No CUDA, no Apple Accelerate, no datacenter, no root.**

Sibling of [`notorch/termux-edition/`](https://github.com/ariannamethod/notorch/tree/main/termux-edition). Mirrors the same approach: upstream `ariannamethod.ai` plus the small Termux delta needed to build and install `aml` / `amlc` / `libaml.a` / `<ariannamethod/ariannamethod.h>` inside `$PREFIX` (`/data/data/com.termux/files/usr`).

Maintained by phone-1 (Defender) and phone-2 (device-2). Neo / Intel do not test this substrate — patches that only affect the Termux build path land here without their review.

## What's verified

- Galaxy A56, Android 15, aarch64-linux-android, 8 GB RAM (phone-1)
- libopenblas 0.3.30 via `pkg install libopenblas` + `pkg-config`
- AML `5/5 build` clean on commit `d4ab1f0` (CPU + BLAS, no CUDA): `libaml.a` 215358 B, `aml` 148288 B, `amlc` 17856 B, header 50912 B
- `readelf -d $PREFIX/bin/aml | grep openblas` → `NEEDED libopenblas.so` confirms BLAS linkage at runtime, not just compile flag
- post-Intel commit `ff7fb97` (backward CPU-sync audit, 16 ops) — on aarch64 Termux without CUDA all `ensure_cpu` calls are no-ops via `#ifdef USE_CUDA`, so zero runtime delta — but install brings the header & lib up to date for any AML organism transpiled here

## Setup

```bash
pkg install -y git build-essential binutils libopenblas pkg-config
ln -sf "$(command -v llvm-ar)" "$PREFIX/bin/ar"   # same as notorch termux-edition

git clone https://github.com/ariannamethod/ariannamethod.ai ~/arianna/ariannamethod.ai
cd ~/arianna/ariannamethod.ai

# Apply the BLAS pkg-config patch (Termux delta — see Makefile.termux.patch in this dir)
patch -p1 < termux-edition/Makefile.termux.patch

make BLAS=1
make BLAS=1 install PREFIX=$PREFIX
amlc --help    # smoke
```

## Termux delta

The single Makefile diff `termux-edition/Makefile.termux.patch` replaces the unconditional `-DUSE_BLAS` / `-lopenblas` in the Linux branch with `pkg-config`-derived flags. Reason: Termux installs `cblas.h` at `$PREFIX/include/cblas.h` but its sibling `openblas_config.h` lives one directory deeper at `$PREFIX/include/openblas/`. Without the extra `-I$PREFIX/include/openblas`, `cblas.h:5: #include "openblas_config.h"` fails with `fatal error: 'openblas_config.h' file not found`.

`pkg-config --cflags openblas` returns the right `-I` flag on both Termux and a standard Debian-style Linux box, so the patch is generic — not Termux-only. Upstream-friendly. Submit upstream when convenient; until then this directory hosts the patch + a `make BLAS=1` workflow that works on phone-1.

## How this differs from notorch termux-edition

- notorch's Termux Makefile already uses `pkg-config` upstream (PR #5). AML doesn't yet — that's why this directory exists.
- notorch ships its own `nt_tape_chuck_step`; AML calls into `libnotorch.a` via Blood runtime when transpiled organism needs an optimizer. AML's own ops are independent of notorch.
- 47/47 notorch tests on aarch64 vs 509/509 AML tests in upstream `make test` — both green on phone-1 substrate post-Intel changes.

## Files in this folder

- `README.md` — this document
- `Makefile.termux.patch` — the BLAS pkg-config patch

The actual code, tests, and examples are in the standard `ariannamethod.ai` tree (`core/`, `runner/`, `tools/`, `tests/`, `examples/`). This folder is the Termux narrative; the implementation is universal.
