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

> **Important caveat — the version-skew tolerance is NOT universal.** The
> forward-compatibility above applies only to the *userspace runtime `.so`
> libraries* we surface on `LD_LIBRARY_PATH` (`libamdhip64.so`, `librocblas.so`,
> …): a `+rocm7.1` wheel `dlopen()`ing 7.2-era runtime `.so`'s is fine. It does
> **not** apply to *device-libs bitcode* — the AMDGCN bitcode the HIP
> compiler/JIT backend links into GPU kernels at codegen time, located via the
> `HIP_DEVICE_LIB_PATH` env var. That is a **different mechanism** and is **not**
> version-compatible across ROCm releases: a 7.2-era device-libs bitcode set
> paired with a `+rocm7.1` torch build **deterministically SIGSEGVs at HIP
> device-init**. `rocmPackages.clr`'s Nix setup-hook leaks `HIP_DEVICE_LIB_PATH`
> (pointing at nixpkgs' newer device-libs) into the shell — we don't compile any
> HIP here, so both `flake.nix` (shellHook) and `devenv.nix` (enterShell)
> **unset** it on entry. If you ever see a segfault at `import torch`/HIP init on
> a host where the install otherwise succeeded, check `echo $HIP_DEVICE_LIB_PATH`
> is empty first (see nix/CLAUDE.md failure-mode #9).

**How the ROCm torch is pinned.** Rather than resolving the dependency graph
live on every shell entry (which was slow and non-deterministic — `torch` would
resolve to a PyPI CUDA build and had to be force-reinstalled afterward), the
Python stack is installed from a committed, fully-pinned + hashed lock file,
`nix/requirements.lock.txt`. In it, `torch`/`torchvision` are pinned to their
`+rocm7.1` builds from the index above, and `huggingface-hub` is pinned into the
`>=1.5.0,<2.0` range that `transformers` requires. `setup-unsloth` (and the
first `devenv shell` entry) install it with a single `uv pip sync`, then swap
`unsloth` to your local checkout with `uv pip install -e . --no-deps`. Verify
the result inside the shell with:

```bash
python -c "import torch; print(torch.__version__, torch.version.hip, torch.cuda.is_available())"
# expect a +rocm7.1 build, torch.version.hip set, and True on a real AMD host
```

**Regenerating the lock.** The lock is NOT regenerated automatically — repeat
shell entries with an unchanged lock skip reinstalling anything (that's the
point). When `studio/backend/requirements/base.txt` or `studio.txt` changes
upstream, regenerate it from inside the shell:

```bash
lock-deps          # rewrites nix/requirements.lock.txt AND both SBOMs
git diff nix/requirements.lock.txt   # review, confirm torch is still +rocm7.1
```

then commit the regenerated `nix/requirements.lock.txt`. If upstream bumps the
required torch version, update the `torch==...+rocm7.1` / `torchvision` pins
inside the `lock-deps` script (`lockDepsCmd` in `devenv.nix`) first.

The same `lock-deps` run also regenerates the two CycloneDX SBOMs
(`nix/sbom-python.cdx.json` and `nix/sbom-nix.cdx.json`) so they stay in sync
with the dependency graph — commit them alongside the lock. See
[`README.md`](./README.md) ("Dependency traceability") for what they are and why.

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

## `HIP_VISIBLE_DEVICES` — defaults to a single GPU (device 0)

This shell sets `HIP_VISIBLE_DEVICES=0`, pinning every HIP process it spawns to
one GPU. On multi-GPU hosts, a native crash inside AMD's HIP runtime stream
teardown has been observed when a process sees more than one GPU, and training
here does not benefit from multi-GPU visibility anyway (data-parallel GPU count
is 1 regardless). So the safe default is one GPU.

This is a **default, not a hard lock**. If you deliberately want all GPUs visible
for your own experimentation, override it *after* entering the shell:

```bash
export HIP_VISIBLE_DEVICES=0,1   # expose both GPUs
# or:  unset HIP_VISIBLE_DEVICES  # expose all
```

(The value is set unconditionally when the shell activates, so your override only
sticks for the current shell — re-entering resets it to `0`.)

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
| `setup-unsloth` | Force a full (re)setup: npm install/build, then `uv pip sync` the locked ROCm PyTorch + Unsloth stack into the managed venv (bypasses the up-to-date marker) |
| `start-backend` | Start the FastAPI backend (`studio/backend/run.py`) |
| `start-frontend` | Start the Vite dev server for the UI |
| `lock-deps` | Regenerate `nix/requirements.lock.txt` from the upstream requirements files **and** rebuild both CycloneDX SBOMs (`nix/sbom-python.cdx.json`, `nix/sbom-nix.cdx.json`) in the same step (maintainers only; commit the results) |

## Troubleshooting

This section is written for a developer who doesn't know Nix or devenv
internals — if you want the full technical root-cause writeup for any of
these (useful if you're debugging something new, or you're an AI coding
agent maintaining this environment), see
[`CLAUDE.md`](./CLAUDE.md)'s failure-mode catalog, which this section
summarizes in plain language.

### "`devenv shell` seems to reinstall everything every time I open it"

It shouldn't — a repeat `devenv shell` entry with nothing changed should
finish in well under a second, not the ~30 seconds a real install takes. If
you're seeing a full reinstall every single time, something is wrong (see
`CLAUDE.md` failure-mode #9's "follow-on subtlety" for the specific bug this
turned out to be once, now fixed). If you *do* expect it — e.g. you just
regenerated the lock with `lock-deps`, or you're on a brand-new checkout —
that's working as intended: a changed lock always triggers a reinstall.

### "Something crashed with 'Segmentation fault'"

This is almost always a ROCm/GPU-runtime problem, not a bug in your Unsloth
code — a segfault from Python usually means something in the compiled
AMD GPU stack (not this repo's Python) hit a wall. Two specific, already-fixed
causes are documented in `CLAUDE.md` (#7, #9) — a stale environment could
still be hitting an old version of one of them. First things to try:

1. Make sure you're on a fresh `devenv shell` (exit and re-enter) — this
   picks up any environment fixes that have landed since you last entered.
2. If it keeps happening, check `echo $HIP_VISIBLE_DEVICES` and
   `echo $HIP_DEVICE_LIB_PATH` inside the shell — the first should print `0`,
   the second should print nothing at all. If either looks different, that's
   worth reporting as a regression rather than working around yourself.
3. If it doesn't match anything in the failure-mode catalog, that's a
   genuinely new issue — worth writing up (what you ran, the exact error,
   `rocminfo` output) rather than guessing at a fix.

### "How do I know if my training run is using the GPU well?"

Check `rocm-smi` (or the Studio UI's GPU monitor) while a job is running. For
a normal single-model training job, expect **one** GPU near 100% utilization
and the others near idle — Unsloth Studio deliberately picks the *minimum*
number of GPUs a job needs (usually one), not "spread across everything
available." A GPU monitor showing your machine's *total* VRAM pool (e.g. 64GB
across two 32GB cards) is showing hardware capacity, not per-job usage — don't
read that as "the job should be using all of it." If a second GPU shows
memory used but low utilization, check whether something *else* (another
inference server, a leftover process) is holding it — that's normal and not
a sign anything here is broken.

### "What's the SBOM for, and do I need to touch it?"

No — it's maintenance tooling, not something you hand-edit. `nix/sbom-python.cdx.json`
and `nix/sbom-nix.cdx.json` are auto-regenerated by the `lock-deps` script
whenever the dependency lock changes, and exist so that after merging in new
upstream commits, someone (or an agent) can answer "what exactly changed in
what we depend on" without reverse-engineering it by hand. See
[`README.md`](./README.md)'s "Dependency traceability" section for more.

### If you're stuck

Try these, in order, before reaching for anything more involved:

1. `setup-unsloth` — force a full reinstall of the Python stack, bypassing
   the "already installed" fast-skip. Fixes most "something's inconsistent"
   states.
2. Remove `nix/.devenv/` entirely and re-run `devenv shell` — the "nuclear
   option," rebuilds everything from scratch. Safe; it's all regenerable
   local state, never anything you'd lose real work from.
3. If neither resolves it and you suspect it's related to a recent upstream
   merge, the `amd-nix-sync-upstream` skill (or, if you're driving this by
   hand, the three agents it wires together — `amd-nix-maintainer`,
   `amd-nix-sbom-auditor`, `amd-nix-validator`) exists specifically to
   diagnose, fix, and hardware-validate this environment after upstream
   changes. See `CLAUDE.md`'s "When this breaks after an upstream merge"
   for what that actually does.

## Linting and Formatting

`ruff` runs as a pre-commit hook; run it manually with:
```bash
ruff check .
```
