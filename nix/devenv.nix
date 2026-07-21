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
    # Python stack. Install the Unsloth requirements FIRST — they pull `unsloth`
    # from PyPI, which drags in an unpinned CUDA torch + torchvision transitively;
    # that CUDA torch is transient and discarded by the final step below.
    uv pip install --python "$VENV_PYTHON" -r studio/backend/requirements/base.txt
    uv pip install --python "$VENV_PYTHON" -r studio/backend/requirements/studio.txt
    uv pip install --python "$VENV_PYTHON" -e .
    # Authoritative, LAST: force the ROCm torch + torchvision from AMD's prebuilt
    # wheel index, overwriting whatever CUDA build the requirements pulled in.
    # torchvision must also come from the ROCm index or plain torchvision drags
    # CUDA torch back in transitively. No PyTorch wheels exist for ROCm 7.2+ yet,
    # so rocm7.1 is the current blessed tag (see DEVENV.md).
    uv pip install --python "$VENV_PYTHON" torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1 --upgrade --force-reinstall
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

  # ── Processes ──────────────────────────────────────────────────────────
  processes.backend.exec = "start-backend";
  processes.frontend.exec = "start-frontend";

  # ── Shell ──────────────────────────────────────────────────────────────
  enterShell = ''
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

    # Install the Python stack into the managed venv if unsloth is missing.
    # Requirements FIRST (they pull a transient CUDA torch/torchvision from PyPI),
    # then force the ROCm torch + torchvision LAST so the final torch is ROCm, not
    # the CUDA build the requirements dragged in transitively.
    if ! "$VENV_PYTHON" -c "import unsloth" &>/dev/null; then
      echo "Initializing backend dependencies into managed venv..."
      uv pip install --python "$VENV_PYTHON" -r "$REPO_ROOT/studio/backend/requirements/base.txt"
      uv pip install --python "$VENV_PYTHON" -r "$REPO_ROOT/studio/backend/requirements/studio.txt"
      uv pip install --python "$VENV_PYTHON" -e "$REPO_ROOT"
      uv pip install --python "$VENV_PYTHON" torch torchvision --index-url https://download.pytorch.org/whl/rocm7.1 --upgrade --force-reinstall
    fi

    echo ""
    echo "Available scripts:"
    echo "  setup-unsloth  - Run the full setup (frontend build + ROCm python stack)"
    echo "  start-backend  - Start the studio backend"
    echo "  start-frontend - Start the studio frontend"
    echo ""
    echo "To start everything at once, run: devenv up"
  '';
}
