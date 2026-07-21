# Unsloth Studio: AMD (ROCm) Development Environment with devenv

This directory provides a [devenv](https://devenv.sh/) development environment
for running Unsloth Studio on **AMD GPUs (ROCm)** on Linux. It is derived from
the NVIDIA-oriented environment under `debug/unsloth.nvidia/`, but it is
**AMD-only**: there is no CUDA/Intel branching here.

## ⚠️ Read this first: this is a best-effort community environment

**NixOS is not on AMD's officially supported OS list.** AMD's ROCm
compatibility matrix covers Ubuntu, RHEL, SLES, Oracle Linux, Debian, and Azure
Linux — **not** NixOS. The `pkgs.rocmPackages.*` set used here is
**community-maintained in nixpkgs, not AMD-certified**, and has a documented
history of version drift and breakage against AMD's own releases.

What this means in practice:

- This environment is a convenience for developers who already run Nix. It is
  **not** a substitute for AMD's supported install paths.
- If something breaks here, **cross-check against Unsloth's official installer
  on an officially-supported distro before assuming it's a code bug**:
  ```bash
  curl -fsSL https://unsloth.ai/install.sh | sh
  ```
  If the official installer works on Ubuntu but this env doesn't, the fault is
  almost certainly in the Nix ROCm runtime, not in Unsloth.

## What Nix does and does not do here

Nix's role is **deliberately minimal**. It supplies:

- generic dev tooling (git, cmake, ninja, ccache, pkg-config, uv, Python,
  Node.js, pre-commit, …), and
- ROCm **runtime** `.so` libraries on `LD_LIBRARY_PATH` so that prebuilt AMD
  PyTorch/bitsandbytes wheels can find them at runtime.

Nix does **not** build, link against, or replace any AMD-shipped ML software.
`torch`, `bitsandbytes`, and `triton` are **not** built by Nix — they are
installed by `uv pip` from AMD's prebuilt wheel index, exactly as Unsloth's
official AMD docs (<https://unsloth.ai/docs/get-started/install/amd>) prescribe.

### Why the `rocm7.1` wheel tag on a possibly-newer ROCm runtime

`setup-unsloth` installs `torch` **and** `torchvision` from:

```
https://download.pytorch.org/whl/rocm7.1
```

This is intentional. Unsloth's current official AMD docs point real users at the
`rocm7.1` wheel index because **no PyTorch wheels exist yet for ROCm 7.2+**. The
`rocm7.1` wheels are forward-compatible enough to run against a 7.2.x runtime,
so this is the current blessed fallback even when the ROCm libraries on
`LD_LIBRARY_PATH` are newer than 7.1. **Revisit this when AMD ships PyTorch
wheels for ROCm 7.2+** and Unsloth's docs move to a newer tag.

**Install order matters.** Installing `unsloth` (via the requirements files)
pulls an unpinned CUDA `torch`/`torchvision` from PyPI transitively, which would
clobber a ROCm torch installed earlier. So `setup-unsloth` installs the
requirements **first**, then force-reinstalls ROCm `torch torchvision` from the
index above as the **final** step (`--force-reinstall`). `torchvision` must come
from the ROCm index too — a plain PyPI `torchvision` drags CUDA `torch` back in.
Verify the result inside the shell with:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
# expect a +rocm build, torch.version.hip set, and True on a real AMD host
```

> This env intentionally does **not** use the `rocm72-torch291` /
> `rocm711-torch2100` (etc.) optional-dependency extras in the repo's
> `pyproject.toml`. Those pull torch/triton wheels from `repo.radeon.com`, a
> different source than what Unsloth's current official docs point real users
> to. Matching the official docs (`download.pytorch.org/whl/rocm7.1`) keeps this
> env aligned with what a user following the official instructions actually gets.

## `HSA_OVERRIDE_GFX_VERSION` — not set here, by design

`HSA_OVERRIDE_GFX_VERSION` tells the ROCm runtime to treat your GPU as a
different `gfxNNNN` ISA target. It is commonly used to run ROCm on cards that
share an ISA with an officially-supported one but aren't themselves on the
support list (e.g. forcing a consumer RDNA card to a nearby supported target).

**This environment does not set it**, because the correct value is
hardware-specific — a wrong value silently produces incorrect results or
crashes. If your card needs it, set it yourself before launching:

```bash
export HSA_OVERRIDE_GFX_VERSION=11.0.0   # example only — use YOUR card's value
```

Determine your actual target with `rocminfo | grep gfx` first, and only override
when you know the ISA you're mapping to.

## Boundary: userspace runtime only (drivers / WSL2 are your responsibility)

This Nix shell provides **userspace ROCm runtime libraries only**. It does
**not**:

- install kernel drivers (`amdgpu`) or set up `/dev/kfd` on the host, or
- replace `scripts/install_rocm_wsl_strixhalo.sh` for WSL2 users.

On WSL2, `scripts/install_rocm_wsl_strixhalo.sh` is a **prerequisite you run
outside this shell** — it handles the WSL2-specific `/dev/dxg` bridge
(`librocdxg`), the ROCm userspace install, and the Windows-side Adrenalin
driver requirement. Run it first; then use this devenv for the Python/Node dev
loop. The shell's startup check warns (but does not fail) if `/dev/kfd` is
absent, since a missing `/dev/kfd` means the host driver layer isn't ready yet.

## Prerequisites

1. **Install Nix**: <https://nixos.org/download.html>
2. **Install devenv** (2.1+):
   ```bash
   nix profile install tarball+https://github.com/cachix/devenv/tarball/latest
   ```
3. **(Optional) direnv** — to auto-enter the shell when you `cd` in.
4. **A working host ROCm driver layer** — see the boundary note above.

## Getting Started

### 1. Enter the environment
From this `nix/` directory:
```bash
devenv shell
```
On first run this downloads dev tooling and the ROCm runtime libraries and
creates a managed Python venv. `DEVICE_TYPE` is exported as `hip`.

### 2. Initial setup
```bash
setup-unsloth
```
Builds the frontend and installs the Python stack (ROCm PyTorch wheel + Unsloth)
into the managed venv.

### 3. Start development services
```bash
devenv up
```
- **Backend**: `http://localhost:8000` (default)
- **Frontend**: `http://localhost:5173` (Vite dev server)

Verify device selection inside the shell:
```bash
echo $DEVICE_TYPE   # -> hip
```

## Available Scripts

| Script | Description |
| :--- | :--- |
| `setup-unsloth` | Full studio setup: npm install/build, then install the ROCm PyTorch wheel + Unsloth into the managed venv |
| `start-backend` | Start the FastAPI backend (`studio/backend/run.py`) |
| `start-frontend` | Start the Vite dev server for the UI |

## Linting and Formatting

`ruff` runs as a pre-commit hook; run it manually with:
```bash
ruff check .
```
