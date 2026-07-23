{ pkgs, lib, config, inputs, ... }:

# AMD-only Unsloth Studio devenv. No CUDA/Intel branching — this environment
# targets ROCm exclusively. See DEVENV.md for why NixOS is not an AMD-supported
# OS and why Nix supplies only ROCm *runtime* libraries (torch/bitsandbytes/
# triton are still installed via uv from AMD's prebuilt wheel index).
#
# Packages sourcing note: the `rocm-flake` input (nix/flake.nix) is the pinned
# source of truth for the ROCm runtime, but the ROCm packages below are taken
# directly from devenv's own `pkgs` rather than spliced out of the flake. The
# flake exports ONLY `devShells.x86_64-linux.default` (no `packages` output), and
# devenv 2.1 has no first-class API to import a foreign flake's devShell
# packages/env; pulling a devShell's buildInputs is fragile and would risk mixing
# two nixpkgs instances. Instead, devenv.yaml pins nixpkgs to the SAME commit the
# flake pins, so these inlined pkgs.rocmPackages.* are byte-identical to the
# flake's devShell — the flake stays authoritative with zero drift, and the
# package set stays decoupled from the flake's internal structure.

let
  rocmLibs = with pkgs.rocmPackages; [
    clr
    rocblas
    hipblas
    rocm-smi
  ];

  # Shared, gated Python-stack installer used by BOTH enterShell and
  # setup-unsloth so the two never drift. Expects REPO_ROOT and VENV_PYTHON to
  # already be set by the caller. Honors $FORCE_INSTALL=1 to bypass the
  # up-to-date marker (setup-unsloth sets it; enterShell does not).
  #
  # This replaces the former 5-step "install then clobber-then-correct" sequence
  # (5 independent `uv pip install` calls that each re-resolved against live
  # PyPI, bouncing fsspec/filelock/huggingface-hub between versions every shell
  # entry). Now it's ONE deterministic `uv pip sync` against the committed,
  # fully-pinned+hashed nix/requirements.lock.txt, then a single `-e . --no-deps`
  # to swap unsloth to the local editable source without perturbing the locked
  # graph. Gated on a content hash of the lock stored under $DEVENV_STATE so
  # repeat shell entries with an unchanged lock skip the install entirely.
  # See nix/CLAUDE.md failure-mode catalog #7/#8.
  installPyStack = ''
    LOCK_FILE="$REPO_ROOT/nix/requirements.lock.txt"
    MARKER="$DEVENV_STATE/.py-deps.lockhash"
    LOCK_HASH="$(sha256sum "$LOCK_FILE" | cut -d' ' -f1)"
    need=0
    [ "''${FORCE_INSTALL:-0}" = "1" ] && need=1
    # This need-check must answer only "is unsloth installed?" — it must NOT
    # `import unsloth`, because enterShell runs mid-splice: devenv interleaves
    # rocmPackages.clr's setup-hook env exports around the enterShell hook's own
    # commands, so at this line the ROCm env is only PARTIALLY assembled (an
    # inconsistent intermediate state, not merely a missing/extra var). Actually
    # importing unsloth initializes the GPU during that incomplete splice and
    # SIGSEGVs — which used to force need=1 and defeat the fast-skip on every 2nd
    # entry. `importlib.util.find_spec` locates the module WITHOUT executing it:
    # no GPU/HIP init, immune to the splice-timing state. (The post-install
    # verification gate below still actually imports torch to confirm the env is
    # correct — but it runs after the splice completes.) See nix/CLAUDE.md #9.
    #
    # `-I` (isolated mode) is REQUIRED for correctness, not just hygiene: enterShell
    # `cd`s to REPO_ROOT before this runs, and REPO_ROOT contains the `unsloth/`
    # source dir + `unsloth.egg-info`. Python puts cwd on sys.path, so WITHOUT -I
    # `find_spec('unsloth')` resolves to that in-tree source and reports "installed"
    # even when unsloth is genuinely absent from the venv — a false positive that
    # wrongly skips reinstalling and leaves a broken venv with an intact marker.
    # -I drops cwd/PYTHONPATH/user-site from sys.path while KEEPING the venv's own
    # site-packages (derived from the interpreter location, unaffected by -I), so
    # the probe checks what's actually installed. (-I also happens to harden GPU
    # init, on top of find_spec never executing the module.)
    "$VENV_PYTHON" -I -c "import importlib.util, sys; sys.exit(0 if importlib.util.find_spec('unsloth') else 1)" >/dev/null 2>&1 || need=1
    { [ -f "$MARKER" ] && [ "$(cat "$MARKER")" = "$LOCK_HASH" ]; } || need=1
    if [ "$need" = "1" ]; then
      echo "Installing locked Python stack (uv pip sync nix/requirements.lock.txt)..."
      # The marker is written ONLY if sync + editable install + verification ALL
      # succeed (the &&-chain below). Without this, a failed sync used to fall
      # through and still write the marker — recording a broken install as
      # "complete" and, worse, leaving the fast-skip path unreachable (the next
      # entry recomputes need=1 via the failing `import unsloth` and just re-runs
      # the same failing sync). No `set -e`/`return`/`exit` here: this snippet is
      # sourced in enterShell (where `exit` would drop the user's shell) but run
      # as a standalone script in setup-unsloth (where `return` is an error), so
      # an &&-chain is the one form that behaves correctly in both.
      #
      # --index-strategy unsafe-best-match is REQUIRED here (not just at compile
      # time): the lock spans PyPI + the ROCm wheel index, and some packages
      # (e.g. certifi) exist on both at different versions. uv's default
      # first-index-wins can't reconcile that and fails with "No solution found";
      # unsafe-best-match lets it pick the locked version across both indexes.
      #
      # --no-deps on the editable keeps every locked version intact instead of
      # letting uv re-resolve/perturb the graph. The verification gate is
      # defense-in-depth against a future lock regression (failure-mode #3/#4/#6):
      # assert torch is a ROCm build and huggingface-hub is in transformers'
      # required >=1.5.0,<2.0 range. Single-line python avoids `-c` indent issues.
      if uv pip sync --python "$VENV_PYTHON" --index-strategy unsafe-best-match "$LOCK_FILE" \
         && uv pip install --python "$VENV_PYTHON" -e "$REPO_ROOT" --no-deps \
         && "$VENV_PYTHON" -c "import torch, huggingface_hub; assert torch.version.hip is not None, f'torch is not a ROCm build: {torch.__version__}'; hf_ver = tuple(int(p) for p in huggingface_hub.__version__.split('.')[:2]); assert (1, 5) <= hf_ver < (2, 0), f'huggingface-hub {huggingface_hub.__version__} does not satisfy >=1.5.0,<2.0'; print(f'[nix/devenv] verified torch={torch.__version__} huggingface_hub={huggingface_hub.__version__}')"; then
        echo "$LOCK_HASH" > "$MARKER"
      else
        echo "ERROR: Python stack install failed — NOT marking as complete. Fix the error above and re-run 'setup-unsloth' (or re-enter the shell)." >&2
      fi
    else
      echo "Python stack already installed and lock unchanged — skipping install."
    fi
  '';

  # Regenerate nix/requirements.lock.txt from the upstream requirements files.
  # Run manually (and commit the result) whenever studio/backend/requirements/*.txt
  # changes upstream — this is an amd-nix-maintainer maintenance task, NOT
  # something that happens automatically on shell entry. See nix/CLAUDE.md and
  # nix/DEVENV.md. The torch/torchvision +rocm7.1 pins in the override are what
  # force those two off the ROCm wheel index (uv's index priority alone does not
  # reliably pick the ROCm build over the PyPI one); bump them here when upstream
  # moves to a newer torch. The huggingface-hub override forces past studio.txt's
  # stale ==0.36.2 exact pin (transformers require_version()s >=1.5.0,<2.0).
  #
  # --no-emit-package triton is load-bearing on AMD: torch's ROCm wheel depends
  # on `triton-rocm` while unsloth/unsloth-zoo/cut-cross-entropy depend on plain
  # `triton`, and BOTH own the `triton/` import path. Shipping both in the lock
  # installs them concurrently with no deterministic winner, yielding a MIXED
  # on-disk layout (e.g. `triton/_utils.py` from triton-rocm 3.6.0 but
  # `triton/experimental/gluon/` from triton 3.7.1) that top-level `import triton`
  # hides but that breaks the first real training kernel with
  # `ImportError: cannot import name 'apply_with_path' from 'triton._utils'`.
  # Excluding plain `triton` leaves `triton-rocm` as the sole provider of the
  # `triton` module — the correct ROCm build. See failure-mode catalog #8.
  lockDepsCmd = ''
    REPO_ROOT="$(cd "$DEVENV_ROOT/.." && pwd)"
    VENV_PYTHON="$DEVENV_STATE/venv/bin/python"
    cd "$REPO_ROOT"
    printf 'huggingface-hub>=1.5.0,<2.0\ntorch==2.10.0+rocm7.1\ntorchvision==0.25.0+rocm7.1\n' > nix/lock-override.txt
    UV_PYTHON_DOWNLOADS=never uv pip compile \
      --python "$VENV_PYTHON" \
      --no-header \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/rocm7.1 \
      --index-strategy unsafe-best-match \
      --override nix/lock-override.txt \
      --no-emit-package unsloth \
      --no-emit-package triton \
      --emit-index-url \
      --generate-hashes \
      --output-file nix/requirements.lock.txt \
      studio/backend/requirements/base.txt studio/backend/requirements/studio.txt
    rm -f nix/lock-override.txt
    echo "Wrote nix/requirements.lock.txt — review and commit it (see nix/CLAUDE.md maintenance notes)."

    # ── SBOMs (CycloneDX) ────────────────────────────────────────────────
    # Regenerating the lock and regenerating the SBOMs are ONE maintenance
    # action (bundled here on purpose, not a second script) so the two never
    # drift: whenever the Python closure changes, both traceability documents
    # are rewritten in the same step. Two independently-generated CycloneDX 1.6
    # JSON docs are emitted — deliberately NOT a hand-merged single file:
    #   nix/sbom-python.cdx.json  — the pip/PyPI dependency closure (this lock)
    #   nix/sbom-nix.cdx.json     — the Nix/system devShell store-path closure
    # Each references the other in its own metadata. See nix/README.md for the
    # what/why and the deliberate two-file split.
    echo "Regenerating SBOMs (CycloneDX)..."

    # Python layer: cyclonedx-py parses the freshly-written lock. Run it
    # EPHEMERALLY via uvx (never a permanent venv/lock dependency — same spirit
    # as this compile step being a one-off maintenance action). --python
    # "$VENV_PYTHON" pins uvx to the nix-built venv interpreter, because uv's own
    # downloaded CPython is a generic-linux binary that cannot run on NixOS.
    # --output-reproducible drops the timestamp and random serialNumber so a
    # regenerated SBOM diffs cleanly across an upstream merge (the entire point
    # of keeping it in git). The index URLs mirror the lock's own emitted
    # directives so component references stay accurate.
    uvx --python "$VENV_PYTHON" --from cyclonedx-bom cyclonedx-py requirements \
      nix/requirements.lock.txt \
      --output-format JSON --output-reproducible --spec-version 1.6 \
      --index-url https://pypi.org/simple \
      --extra-index-url https://download.pytorch.org/whl/rocm7.1 \
      -o nix/sbom-python.cdx.json

    # Point the Python doc at its Nix-layer sibling (cyclonedx-py can't emit
    # custom metadata properties itself). Idempotent: re-running never dupes.
    "$VENV_PYTHON" - nix/sbom-python.cdx.json <<'PY_XREF'
import json, sys
p = sys.argv[1]
d = json.load(open(p))
props = d.setdefault("metadata", {}).setdefault("properties", [])
have = {x.get("name") for x in props}
for name, value in (("unsloth:sbom-layer", "python"),
                    ("unsloth:sibling-sbom", "nix/sbom-nix.cdx.json")):
    if name not in have:
        props.append({"name": name, "value": value})
json.dump(d, open(p, "w"), indent=2, ensure_ascii=False)
open(p, "a").write("\n")
PY_XREF

    # Nix/system layer: no standard nix->CycloneDX tool exists, so derive it
    # in-repo. Enumerate the devShell's full transitive store-path closure and
    # emit one component per path — name/version parsed from the store-path
    # basename (path-info's own "version" field is the JSON schema version, not
    # the package version), exact provenance carried as nix:store-path /
    # nix:output-hash properties, and the nixpkgs commit (the root of trust for
    # this layer) read from flake.lock and recorded as BOM metadata. The
    # generator is written to a temp file so path-info can stream to it on stdin.
    GEN_NIX_SBOM="$DEVENV_STATE/nix-sbom-gen.py"
    cat > "$GEN_NIX_SBOM" <<'PY_NIX_SBOM'
import json, sys, re

# Reads `nix path-info -r --json` from stdin; argv: <flake.lock> <output.json>.
flake_lock_path = sys.argv[1]
out_path = sys.argv[2]

paths = json.load(sys.stdin)

with open(flake_lock_path) as f:
    lock = json.load(f)
nixpkgs_commit = lock["nodes"]["nixpkgs"]["locked"]["rev"]

STORE_RE = re.compile(r"^/nix/store/([a-z0-9]{32})-(.+)$")

# Trailing tokens that denote a Nix *output* (a build product of the same
# package) rather than part of the version. One store path == one output.
KNOWN_OUTPUTS = {
    "dev", "bin", "lib", "man", "doc", "devdoc", "out", "info", "static",
    "debug", "dist", "devman", "npm", "corepack", "py",
}

def parse_name_version(basename):
    # nixpkgs names store paths <pname>-<version>[-<output>]. Peel a trailing
    # known-output token, then treat the first hyphen-token starting with a
    # digit as the version start; everything before it is the name.
    tokens = basename.split("-")
    output = None
    if len(tokens) > 1 and tokens[-1] in KNOWN_OUTPUTS:
        output = tokens.pop()
    ver_idx = next((i for i, t in enumerate(tokens) if t and t[0].isdigit()), None)
    if ver_idx is None or ver_idx == 0:
        return (basename if output is None else "-".join(tokens)), None, output
    return "-".join(tokens[:ver_idx]), "-".join(tokens[ver_idx:]), output

components = []
for store_path in sorted(paths):
    info = paths[store_path]
    m = STORE_RE.match(store_path)
    if not m:
        continue
    out_hash, basename = m.group(1), m.group(2)
    name, version, output = parse_name_version(basename)
    props = [
        {"name": "nix:store-path", "value": store_path},
        {"name": "nix:output-hash", "value": out_hash},
    ]
    if info.get("narHash"):
        props.append({"name": "nix:nar-hash", "value": info["narHash"]})
    if output:
        props.append({"name": "nix:output", "value": output})
    comp = {"type": "library", "bom-ref": store_path, "name": name, "properties": props}
    if version:
        comp["version"] = version
    components.append(comp)

bom = {
    "bomFormat": "CycloneDX",
    "specVersion": "1.6",
    "version": 1,
    "metadata": {
        "component": {
            "type": "application",
            "bom-ref": "unsloth-amd-devenv-shell",
            "name": "unsloth-studio-amd-devenv-shell",
            "description": (
                "Nix/system layer of the Unsloth Studio AMD/ROCm devenv: the "
                "transitive store-path closure of the flake's devShell (dev "
                "tooling + ROCm runtime libraries). The Python ML stack is NOT "
                "here - see the sibling SBOM nix/sbom-python.cdx.json."
            ),
        },
        "properties": [
            {"name": "nix:nixpkgs-commit", "value": nixpkgs_commit},
            {"name": "nix:flake-ref", "value": "github:NixOS/nixpkgs/" + nixpkgs_commit},
            {"name": "unsloth:sbom-layer", "value": "nix-system"},
            {"name": "unsloth:sibling-sbom", "value": "nix/sbom-python.cdx.json"},
            {"name": "unsloth:purl-note",
             "value": ("no standard pkg:nix PackageURL scheme exists; components "
                       "carry no purl and are identified by name+version plus the "
                       "nix:store-path / nix:output-hash properties (the exact "
                       "provenance) instead")},
        ],
        "tools": {"components": [
            {"type": "application", "name": "nix path-info",
             "description": "closure enumeration (nix path-info -r --json)"},
            {"type": "application", "name": "unsloth-nix-sbom",
             "description": "in-repo Nix->CycloneDX generator, run by the lock-deps devenv script"},
        ]},
    },
    "components": components,
}

with open(out_path, "w") as f:
    json.dump(bom, f, indent=2, ensure_ascii=False)
    f.write("\n")

print("wrote", out_path, "with", len(components), "components")
PY_NIX_SBOM
    nix path-info -r --json "$REPO_ROOT/nix#devShells.x86_64-linux.default" \
      | "$VENV_PYTHON" "$GEN_NIX_SBOM" nix/flake.lock nix/sbom-nix.cdx.json
    rm -f "$GEN_NIX_SBOM"

    echo "Wrote nix/sbom-python.cdx.json and nix/sbom-nix.cdx.json — review and commit them (see nix/README.md)."
  '';
in {
  # ── Packages ───────────────────────────────────────────────────────────
  packages = with pkgs; [
    git
    cmake
    ninja
    ccache
    pkg-config
    openssl
    zlib
    curl
    uv
    python312
    nodejs_22
    pre-commit
    zstd                    # provides libzstd.so.1 that the ROCm torch C ext dlopens
    rocmPackages.rocminfo   # host GPU/ISA enumeration for the shell banner
  ] ++ rocmLibs;

  # ── Languages ──────────────────────────────────────────────────────────
  # Toolchains are managed manually to avoid failing root-level automatic syncs.
  languages.python = {
    enable = true;
    venv.enable = true; # Creates $DEVENV_STATE/venv
  };
  languages.javascript.enable = false;

  # ── Environment ────────────────────────────────────────────────────────
  env = {
    # Remote inference endpoint. If set, LlamaCppBackend proxies to this URL
    # instead of starting a local llama-server.
    LLAMA_SERVER_URL = "";

    # ROCm runtime discovery. clr provides the HIP runtime (libamdhip64);
    # ROCM_PATH/HIP_PATH point ML tooling at it. NVIDIA env vars are
    # intentionally absent. HSA_OVERRIDE_GFX_VERSION is intentionally NOT set
    # here — it is hardware-specific; set it yourself if your card needs it
    # (see DEVENV.md).
    ROCM_PATH = "${pkgs.rocmPackages.clr}";
    HIP_PATH = "${pkgs.rocmPackages.clr}";
    # pkgs.zstd supplies libzstd.so.1, which torch's ROCm C extension dlopens at
    # import (without it: "ImportError: libzstd.so.1: cannot open shared object file").
    LD_LIBRARY_PATH = lib.makeLibraryPath (rocmLibs ++ [ pkgs.zstd "/run/opengl-driver" ]);

    # Pin every HIP process spawned in this shell to a single GPU. This host has
    # 2 visible AMD GPUs, and a native SIGSEGV inside libamdhip64's stream
    # teardown (hip::Device::NullStream() -> HostQueue::terminate() ->
    # ReferenceCountedObject::release()) has been hit in two independent code
    # paths whenever both GPUs are visible to a process: the training subprocess
    # (since fixed in studio/backend/core/training/worker.py) and a bare
    # `import unsloth` in the enterShell guard below. Training never benefits
    # from multi-GPU visibility here anyway (Data Parallel GPUs = 1 even with 2
    # visible), so default the whole shell to one GPU rather than patching each
    # call site. Kept in parity with flake.nix's mkShell env. This is a default,
    # not a hard lock: a user who wants both GPUs for their own experimentation
    # can `export HIP_VISIBLE_DEVICES=0,1` (or unset it) after entering the
    # shell. See nix/CLAUDE.md failure-mode #7 and DEVENV.md.
    HIP_VISIBLE_DEVICES = "0";
  };

  # ── Git hooks ──────────────────────────────────────────────────────────
  git-hooks.hooks.ruff.enable = true;

  # ── Scripts ────────────────────────────────────────────────────────────
  # devenv.nix lives one level below the repo root (nix/), so all studio/... paths
  # must be anchored to the repo root, not to cwd. $DEVENV_ROOT is the directory
  # holding devenv.nix (nix/); its parent is the repo root. uv pip install is
  # given the venv interpreter explicitly (--python): languages.python.venv sets
  # up $DEVENV_STATE/venv but uv keys off VIRTUAL_ENV/--python, not
  # UV_PROJECT_ENVIRONMENT, so without this it warns "No virtual environment found".
  scripts."setup-unsloth".exec = ''
    echo "Running Unsloth Studio setup (AMD/ROCm)..."
    REPO_ROOT="$(cd "$DEVENV_ROOT/.." && pwd)"
    VENV_PYTHON="$DEVENV_STATE/venv/bin/python"
    cd "$REPO_ROOT"
    # Ensure frontend is built
    (cd studio/frontend && { [ -d node_modules ] || npm install; } && npm run build)
    # Python stack: single deterministic `uv pip sync` of the committed lock,
    # then swap unsloth to the local editable source. setup-unsloth is the
    # explicit "(re)do the setup" command, so it forces a reinstall regardless of
    # the up-to-date marker; enterShell (below) honors the marker to stay fast.
    FORCE_INSTALL=1
    ${installPyStack}
  '';

  scripts."start-backend".exec = ''
    echo "Starting Unsloth Studio Backend..."
    cd "$(cd "$DEVENV_ROOT/.." && pwd)"
    python studio/backend/run.py
  '';

  scripts."start-frontend".exec = ''
    echo "Starting Unsloth Studio Frontend..."
    cd "$(cd "$DEVENV_ROOT/.." && pwd)/studio/frontend"
    if [ ! -d "node_modules" ]; then npm install; fi
    npm run dev
  '';

  # Regenerate nix/requirements.lock.txt after upstream requirements changes.
  # Manual + commit the result; see lockDepsCmd's comment and nix/CLAUDE.md.
  scripts."lock-deps".exec = lockDepsCmd;

  # ── Processes ──────────────────────────────────────────────────────────
  processes.backend.exec = "start-backend";
  processes.frontend.exec = "start-frontend";

  # ── Shell ──────────────────────────────────────────────────────────────
  enterShell = ''
    # rocmPackages.clr's Nix setup-hook exports HIP_DEVICE_LIB_PATH at nixpkgs'
    # rocm-device-libs bitcode (7.2-era). We compile no HIP code here (only
    # dlopen runtime .so's for the pip-installed +rocm7.1 torch), and a
    # device-libs *bitcode* version skew (7.2 bitcode vs the wheel's bundled 7.1)
    # deterministically SIGSEGVs at HIP device-init — a DIFFERENT mechanism from
    # the LD_LIBRARY_PATH runtime-.so skew, which IS fine. enterShell runs after
    # the setup-hook, so this unset reliably clears the leak. Kept in parity with
    # flake.nix's shellHook. See DEVENV.md and nix/CLAUDE.md failure-mode #9.
    unset HIP_DEVICE_LIB_PATH

    echo "╔══════════════════════════════════════╗"
    echo "║   Unsloth Dev Shell (AMD / ROCm)     ║"
    echo "╚══════════════════════════════════════╝"
    echo ""

    # AMD runtime check (warn, don't fail). /dev/kfd is the ROCm kernel
    # interface; absence means the host driver layer isn't ready (WSL2 users:
    # run scripts/install_rocm_wsl_strixhalo.sh first — see DEVENV.md).
    if [ ! -e /dev/kfd ]; then
      echo "⚠️  /dev/kfd not found — AMD ROCm kernel interface unavailable."
      echo "    The GPU won't be visible until the host driver layer is ready."
      echo "    WSL2: run scripts/install_rocm_wsl_strixhalo.sh first."
    else
      echo "✅ ROCm kernel interface present (/dev/kfd)."
      if command -v rocminfo >/dev/null 2>&1; then
        gfx=$(rocminfo 2>/dev/null | grep -oE 'gfx[0-9a-z]+' | grep -v 'gfx000' | head -1 || true)
        [ -n "$gfx" ] && echo "    GPU ISA target: $gfx"
      fi
    fi
    # Always hip: this environment is AMD-only and feeds unsloth/device_type.py.
    export DEVICE_TYPE="hip"
    echo ""

    # Point uv at the managed venv. UV_PROJECT_ENVIRONMENT helps `uv run`; the
    # explicit --python below is what makes `uv pip install` target the venv.
    export UV_PROJECT_ENVIRONMENT="$DEVENV_STATE/venv"

    # devenv.nix is at nix/, so anchor studio/... paths to the repo root
    # ($DEVENV_ROOT is nix/; its parent is the repo root) rather than cwd.
    REPO_ROOT="$(cd "$DEVENV_ROOT/.." && pwd)"
    VENV_PYTHON="$DEVENV_STATE/venv/bin/python"

    # Location-aware bootstrap.
    if [ ! -f "$REPO_ROOT/studio/frontend/node_modules/.bin/vite" ]; then
      echo "Initializing frontend dependencies..."
      (cd "$REPO_ROOT/studio/frontend" && npm install)
    fi

    # Install/refresh the Python stack. installPyStack is gated: it runs only
    # when unsloth is missing OR the committed lock's content hash differs from
    # the marker under $DEVENV_STATE, so a repeat `devenv shell` with an
    # unchanged lock skips reinstalling anything. (FORCE_INSTALL is unset here,
    # so the marker is honored — setup-unsloth sets it to force a reinstall.)
    ${installPyStack}

    echo ""
    echo "Available scripts:"
    echo "  setup-unsloth  - Force a full (re)setup (frontend build + ROCm python stack)"
    echo "  start-backend  - Start the studio backend"
    echo "  start-frontend - Start the studio frontend"
    echo "  lock-deps      - Regenerate nix/requirements.lock.txt (maintainers; commit result)"
    echo ""
    echo "To start everything at once, run: devenv up"
  '';
}
