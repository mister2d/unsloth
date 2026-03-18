{ pkgs, lib, config, ... }:

let
  # Detect GPU at build/eval time (for package selection)
  # Note: This works best if you re-evaluate on the target machine
  hasNvidia = builtins.pathExists "/dev/nvidia0";
  hasAmd = builtins.pathExists "/dev/kfd";
  hasIntel = builtins.pathExists "/dev/dri/renderD128";

  # Common packages for all GPU types
  common-pkgs = with pkgs; [
    git
    cmake
    ninja
    ccache
    pkg-config
    openssl
    zlib
    curl
  ];

  # GPU-specific packages
  gpu-pkgs = 
    if hasNvidia then [
      pkgs.cudaPackages.cudatoolkit
      pkgs.cudaPackages.cudnn
      pkgs.cudaPackages.libcublas
      pkgs.cudaPackages.cuda_nvcc
    ] else if hasAmd then [
      pkgs.rocmPackages.clr
      pkgs.rocmPackages.rocblas
      pkgs.rocmPackages.hip-common
      pkgs.rocmPackages.hipcc
      pkgs.rocmPackages.rocm-smi
    ] else [ ];

in {
  # ── Packages ───────────────────────────────────────────────────────────
  packages = common-pkgs ++ gpu-pkgs ++ [ 
    pkgs.uv 
    pkgs.python312
    pkgs.nodejs_22
    pkgs.pre-commit
  ];

  # ── Languages ──────────────────────────────────────────────────────────
  # We manage toolchains manually to avoid failing root-level automatic syncs
  languages.python = {
    enable = true;
    venv.enable = true; # Creates $DEVENV_STATE/venv
  };
  languages.javascript.enable = false;

  # ── Environment ────────────────────────────────────────────────────────
  env = {
    # Specify the remote endpoint for inference
    # If set, LlamaCppBackend will use this URL instead of starting a local server
    LLAMA_SERVER_URL = ""; 

    # GPU-related environment variables
    CUDA_PATH = if hasNvidia then "${pkgs.cudaPackages.cudatoolkit}" else "";
    LD_LIBRARY_PATH = lib.makeLibraryPath (
      (if hasNvidia then [ 
        pkgs.cudaPackages.cudatoolkit 
        pkgs.cudaPackages.cudnn 
        pkgs.cudaPackages.libcublas 
        "/run/opengl-driver"
      ] else if hasAmd then [
        pkgs.rocmPackages.clr
        pkgs.rocmPackages.rocblas
        "/run/opengl-driver"
      ] else [ ])
    );
  };

  # ── Scripts ────────────────────────────────────────────────────────────
  scripts."setup-unsloth".exec = ''
    echo "Running Unsloth Studio setup..."
    # Ensure frontend is built
    cd studio/frontend
    if [ ! -d "node_modules" ]; then npm install; fi
    npm run build
    cd ../..
    # Backend setup
    uv pip install -r studio/backend/requirements/base.txt
    uv pip install -r studio/backend/requirements/studio.txt
    uv pip install -e .
  '';

  scripts."start-backend".exec = ''
    echo "Starting Unsloth Studio Backend..."
    python studio/backend/run.py
  '';

  scripts."start-frontend".exec = ''
    echo "Starting Unsloth Studio Frontend..."
    cd studio/frontend
    if [ ! -d "node_modules" ]; then npm install; fi
    npm run dev
  '';

  # ── Processes ──────────────────────────────────────────────────────────
  processes.backend.exec = "start-backend";
  processes.frontend.exec = "start-frontend";

  # ── Shell ──────────────────────────────────────────────────────────────
  enterShell = ''
    echo "╔══════════════════════════════════════╗"
    echo "║     Unsloth Development Shell        ║"
    echo "╚══════════════════════════════════════╝"
    echo ""
    echo "GPU Detection:"
    if [ -c /dev/nvidia0 ]; then
      echo "✅ NVIDIA GPU detected"
      export DEVICE_TYPE="cuda"
    elif [ -c /dev/kfd ]; then
      echo "✅ AMD GPU detected"
      export DEVICE_TYPE="hip"
    else
      echo "ℹ️  No supported GPU detected (CPU mode)"
      export DEVICE_TYPE="cpu"
    fi
    echo ""

    # Ensure UV knows where the venv is for automatic targeting
    export UV_PROJECT_ENVIRONMENT="$DEVENV_STATE/venv"

    # Location-aware initialization
    if [ ! -f "studio/frontend/node_modules/.bin/vite" ]; then
      echo "Initializing frontend dependencies..."
      (cd studio/frontend && npm install)
    fi

    # Check if unsloth is installed in the venv
    if ! python -c "import unsloth" &>/dev/null; then
      echo "Initializing backend dependencies into managed venv..."
      uv pip install -r studio/backend/requirements/base.txt
      uv pip install -r studio/backend/requirements/studio.txt
      uv pip install -e .
    fi

    echo ""
    echo "Available scripts:"
    echo "  setup-unsloth  - Run the full setup"
    echo "  start-backend  - Start the studio backend"
    echo "  start-frontend - Start the studio frontend"
    echo ""
    echo "To start everything at once, run: devenv up"
  '';
}
