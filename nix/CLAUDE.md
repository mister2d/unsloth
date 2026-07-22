# nix/ — AMD/ROCm Nix devShell + devenv for Unsloth Studio

## What this is

A Nix flake (`flake.nix`) plus a devenv 2.x wrapper (`devenv.yaml`, `devenv.nix`) providing an AMD-only reproducible dev environment for Unsloth Studio, sibling to the NVIDIA-oriented `debug/unsloth.nvidia/` (which uses plain devenv.sh with eval-time hardware branching, no flake).

## Hard architectural constraint — read this before changing anything

**AMD does not officially support NixOS.** AMD's ROCm compatibility matrix only lists Ubuntu, RHEL, SLES, Oracle Linux, Debian, and Azure Linux. `pkgs.rocmPackages.*` in nixpkgs is community-maintained, not AMD-built or certified, and has a documented history of drift/breakage relative to AMD's actual releases. Unsloth's own official AMD install docs use a one-line installer that `uv pip install`s AMD's official prebuilt PyTorch/bitsandbytes wheels from `download.pytorch.org/whl/rocmX.Y` — no Nix involved on their side.

Given that, **Nix's role here is deliberately minimal**: it supplies only generic dev tooling (git, cmake, ninja, python, node, uv) plus the minimal ROCm *runtime* shared libraries on `LD_LIBRARY_PATH` (`clr`, `rocblas`, `hipblas`, `rocm-smi`, `rocminfo`, `zstd`) so that AMD's own official prebuilt wheels can `dlopen()` `libamdhip64.so`/`librocblas.so`/`libzstd.so.1` etc. at runtime. Nix never builds, links against, or replaces any AMD-shipped software — it exists purely to fill the FHS-shaped hole NixOS otherwise leaves open for prebuilt manylinux wheels (see: `nix flake check`/`nix develop` don't require a real AMD GPU to succeed, because there's no eval-time hardware branching — this environment is AMD-only by construction).

Do not "improve" this by trying to nix-build torch/bitsandbytes/triton, and do not switch the Python install source away from `https://download.pytorch.org/whl/rocm7.1` (see failure-mode catalog below for why the pinned wheel tag matters) without a good reason and updating this doc.

## File map

- `flake.nix` / `flake.lock` — the AMD devShell, independently `nix flake check`/`nix develop`-able. Owns: nixpkgs pin, package list, `LD_LIBRARY_PATH`/`ROCM_PATH`/`HIP_PATH` construction, the runtime `/dev/kfd` presence shellHook banner.
- `devenv.yaml` / `devenv.lock` — devenv's input pins; consumes `flake.nix` via a `path:./` input.
- `devenv.nix` — orchestration: inlines the same `rocmPackages.*` set as the flake (kept in parity — the flake only exports `devShells`, not a `packages` output, so devenv.nix can't cleanly splice from it), plus `enterShell`/`scripts` that install the Python ML stack via `uv pip`, run the frontend build, and start backend/frontend processes.
- `DEVENV.md` — user-facing usage docs (prerequisites, `devenv shell` → `setup-unsloth` → `devenv up`, scripts table, troubleshooting).
- `.gitignore` — ignores devenv/direnv generated state and the per-user `.amd-validator-host` file (see below). `devenv.lock`/`flake.lock` ARE tracked (pinned inputs).

## Failure-mode catalog (symptom → root cause → fix)

Every entry here was a real bug hit and fixed during development or maintenance of this environment. Check here first before re-diagnosing from scratch.

1. **`cd: studio/frontend: No such file or directory` / `File not found: studio/backend/requirements/base.txt`** — `devenv.nix` lives at `nix/devenv.nix`, one directory below the repo root, but scripts referenced `studio/...` as if cwd were the repo root (copied from `debug/unsloth.nvidia/devenv.nix`, which genuinely does sit at the repo root). Fix: every script anchors via `REPO_ROOT="$(cd "$DEVENV_ROOT/.." && pwd)"` and references `"$REPO_ROOT/studio/..."`, never bare relative paths.
2. **`uv pip install` warns "No virtual environment found; run 'uv venv'..."** even with `languages.python.venv.enable = true` — `uv pip install` keys off `VIRTUAL_ENV`/`--python`, not the `UV_PROJECT_ENVIRONMENT` env var that `enterShell` sets. Fix: every `uv pip install` call passes `--python "$VENV_PYTHON"` explicitly (`VENV_PYTHON="$DEVENV_STATE/venv/bin/python"`).
3. **Final `torch` ends up as a PyPI CUDA build (`torch==2.12.1+cu130`), `torch.cuda.is_available()` False, `torch.version.hip` None** — `studio/backend/requirements/base.txt` pulls `unsloth`/`unsloth-zoo` transitively, which resolves an unpinned `torch`/`torchvision` from PyPI (CUDA build) with no `--index-url` override. Installing the ROCm torch wheel *before* the requirements files doesn't help — the requirements install clobbers it right back. Fix: requirements (`base.txt`, `studio.txt`, `-e .`) run FIRST (their transient CUDA torch is discarded), then torch **and** torchvision are force-reinstalled LAST from `https://download.pytorch.org/whl/rocm7.1` with `--upgrade --force-reinstall` (torchvision must be included too — a plain PyPI torchvision alone drags CUDA torch back in transitively).
4. **`Failed to import ML libraries: huggingface-hub>=1.5.0,<2.0 is required ... but found huggingface-hub==0.36.2`** (and/or a non-fatal `cannot import name 'is_offline_mode' from 'huggingface_hub'` warning at startup) — same failure class as #3, different package. `studio/backend/requirements/studio.txt` pins `huggingface-hub==0.36.2` (a stale exact pin), which runs *after* `base.txt` resolves a modern, compatible `huggingface-hub` and downgrades it back down. `transformers` enforces `require_version("huggingface-hub>=1.5.0,<2.0")` at import time. Fix (applied): same clobber-then-correct pattern as torch — in **both** install blocks (`enterShell` and `scripts."setup-unsloth"`), immediately after the final torch/torchvision force-reinstall, `uv pip install --python "$VENV_PYTHON" "huggingface-hub>=1.5.0,<2.0" --upgrade` runs, followed by the #6 verification gate.
5. **`ImportError: libzstd.so.1: cannot open shared object file`** on `import torch` — torch's ROCm build `dlopen()`s `libzstd.so.1`, which wasn't on `LD_LIBRARY_PATH`. Fix: `pkgs.zstd` added to `packages` and to the `LD_LIBRARY_PATH` `lib.makeLibraryPath [...]` list in **both** `flake.nix` and `devenv.nix` — kept in parity, same as every other runtime lib.
6. **Post-install verification gate** — because #3 and #4 are the same failure class (an intermediate requirements install silently clobbering an authoritative package version) hitting twice, the final install steps are followed by an assertion (via the venv python) that `torch.version.hip is not None` and `huggingface_hub.__version__` falls in the required range, printing a confirmation line on success and failing loudly (non-zero exit, at `setup-unsloth`/`enterShell` time) otherwise — so a *third* occurrence of this pattern surfaces immediately instead of mid-training-run.

If you hit a **new** variant of "an intermediate install step clobbers an authoritative package," the fix pattern is: (a) let the clobber happen, (b) correct it explicitly as the last step, (c) extend the verification gate to assert the corrected state. Don't try to prevent the clobber by reordering `base.txt`/`studio.txt`/`-e .` relative to each other — their own internal ordering isn't under our control and isn't the actual problem; the problem is always "the environment must end up correct regardless of what those files transitively pull."

## Real-hardware validation target — never hardcoded

This repo/branch is shared across users and hosts, so no specific SSH hostname is ever committed here or in the agent definitions. `amd-nix-validator` resolves its target host per-run, in order: a gitignored `nix/.amd-validator-host` file (`user@hostname`, one line) if present, else the `AMD_NIX_VALIDATOR_HOST` environment variable, else it asks. If you're setting up your own AMD box for validation, point at it with:

```
echo youruser@yourhost > nix/.amd-validator-host
```

(already covered by `nix/.gitignore` — never commit this file with a real value in it).

Also note: the GPU ISA target reported by `rocminfo` has varied between sessions on the *same* hostname in this project's history (`gfx1030` vs `gfx1201` observed) — never hardcode an expected ISA/gfx string anywhere in checks or docs; always read live `rocminfo` output.

## When this breaks after an upstream merge

Unsloth Studio is under continuous development; `studio/backend/requirements/*.txt`, `pyproject.toml`, and hardware-detection code all change frequently, and any of those changes can reintroduce a variant of the failure modes above. The repeatable procedure:

1. Reproduce the failure (dispatch `amd-nix-validator` if you don't have direct real-hardware access yourself).
2. Diagnose against the failure-mode catalog above first — most drift breakage repeats an already-seen category.
3. Dispatch `amd-nix-maintainer` to patch `nix/flake.nix`/`nix/devenv.nix`.
4. Dispatch `amd-nix-validator` to confirm on real hardware.
5. Update this catalog with anything new (symptom → root cause → fix), so the next incident resolves faster.

## Non-goals

Do not edit `studio/backend/requirements/*.txt` or `pyproject.toml` to fix an AMD-only version conflict. Those files are shared with `debug/unsloth.nvidia/` and the standard cross-platform installer (`install.sh`); changing them to work around a problem that only manifests in this Nix environment risks breaking those other paths. Absorb and correct the conflict inside `nix/devenv.nix`'s own install steps instead (see the catalog above).

## Tooling used to build/maintain this, if available in your environment

This environment was originally authored using a `nix-flake-architect` skill (for `flake.nix` structure/conventions), a `devenv2-environment-generator` skill (for `devenv.yaml`/`devenv.nix` structure), and the `mcp__nixos-tools__nix` MCP tool (to verify `rocmPackages.*` attribute names/versions against the actual pinned nixpkgs channel before adding them). None of these are guaranteed to be present in every session — `amd-nix-maintainer` uses them when available and falls back to manual verification when not (see its own definition in `.claude/agents/amd-nix-maintainer.md`).
