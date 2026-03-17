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
      pkgs.rocmPackages.hip
      pkgs.rocmPackages.rocm-smi
    ] else [ ];

in {
  # ── Packages ───────────────────────────────────────────────────────────
  packages = common-pkgs ++ (with pkgs; [
    # Node.js for frontend
    nodejs_20
  ]);

  # ── Languages ──────────────────────────────────────────────────────────
  languages.python = {
    enable = true;
    uv.enable = true;
    venv.enable = true;
    lsp.enable = true;
  };

  languages.javascript = {
    enable = true;
    npm.enable = true;
  };

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
    ./studio/setup.sh
  '';

  scripts."start-backend".exec = ''
    echo "Starting Unsloth Studio Backend..."
    # Ensure dependencies are installed
    if [ ! -d "$HOME/.unsloth/studio/.venv" ]; then
      ./studio/setup.sh
    fi
    # Use the venv created by setup.sh or the devenv venv
    # studio/setup.sh creates one in ~/.unsloth/studio/.venv
    source "$HOME/.unsloth/studio/.venv/bin/activate"
    python studio/backend/run.py
  '';

  scripts."start-frontend".exec = ''
    echo "Starting Unsloth Studio Frontend..."
    cd studio/frontend
    if [ ! -d "node_modules" ]; then
      echo "node_modules not found, installing..."
      npm install
    fi
    npm run dev
  '';

  # ── Processes ──────────────────────────────────────────────────────────
  processes.backend.exec = "devenv shell start-backend";
  processes.frontend.exec = "devenv shell start-frontend";

  # ── Git hooks ──────────────────────────────────────────────────────────
  git-hooks.hooks = {
    ruff.enable = true;
    # Add other hooks like prettier for frontend if needed
  };

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
    echo "Available scripts:"
    echo "  setup-unsloth  - Run the official setup script"
    echo "  start-backend  - Start the studio backend"
    echo "  start-frontend - Start the studio frontend"
    echo ""
    echo "To start everything at once, run: devenv up"
  '';
}
