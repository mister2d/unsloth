# Unsloth Studio: Development Environment with devenv

This project uses [devenv](https://devenv.sh/) to provide a reproducible, multi-architecture development environment. It supports **NVIDIA (CUDA)**, **AMD (ROCm)**, and **Intel (XPU)** GPU environments automatically.

## Prerequisites

1.  **Install Nix**: [Follow the official guide](https://nixos.org/download.html).
2.  **Install devenv**:
    ```bash
    nix profile install tarball+https://github.com/cachix/devenv/tarball/latest
    ```
3.  **(Optional) Install direnv**: To automatically enter the shell when you `cd` into the directory.

## Getting Started

### 1. Enter the Environment
Run the following command to enter the development shell. On the first run, this will download and configure all dependencies (Python 3.12, Node.js, GPU libraries, etc.).

```bash
devenv shell
```

### 2. Initial Setup
Inside the shell, run the project's official setup script to build the frontend and configure the Python virtual environment:

```bash
setup-unsloth
```

### 3. Start Development Services
To start both the backend and frontend at once:

```bash
devenv up
```

- **Backend**: Starts on `http://localhost:8000` (by default)
- **Frontend**: Starts on `http://localhost:5173` (Vite dev server)

## GPU Support

The environment automatically detects your hardware and configures `LD_LIBRARY_PATH` and other essential environment variables:

- **NVIDIA**: Detects `/dev/nvidia0` and loads `cudaPackages`.
- **AMD**: Detects `/dev/kfd` and loads `rocmPackages`.
- **Intel**: Detects `/dev/dri/renderD128` and configures for XPU.

You can verify detection inside the shell:
```bash
echo $DEVICE_TYPE
# Output: cuda, hip, or cpu
```

## Remote Inference Configuration

If you are running the Web UI on a device with minimal resources (e.g., CPU-only laptop) and want to use a remote `llama.cpp` server for inference, set the `LLAMA_SERVER_URL` environment variable.

### Usage:
1.  **Start your remote llama.cpp server**:
    ```bash
    ./llama-server -m your-model.gguf --host 0.0.0.0 --port 8080
    ```
2.  **Configure Unsloth Studio to use it**:
    ```bash
    export LLAMA_SERVER_URL="http://your-remote-ip:8080"
    unsloth studio
    ```

When `LLAMA_SERVER_URL` is set, the studio backend will skip starting a local `llama-server` subprocess and proxy all GGUF inference requests to the remote endpoint.

## Available Scripts

The following scripts are available directly in the `devenv` shell:

| Script | Description |
| :--- | :--- |
| `setup-unsloth` | Runs the full studio setup (npm install, build, python venv) |
| `start-backend` | Starts the FastAPI backend with hot-reload |
| `start-frontend` | Starts the Vite development server for the UI |

## Linting and Formatting

The environment includes `ruff` as a pre-commit hook. To run it manually:
```bash
ruff check .
```
