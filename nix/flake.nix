{
  description = "Unsloth Studio dev shell for AMD/ROCm GPU development";

  # nixpkgs is pinned to the exact nixos-unstable commit already locked in the
  # repo's NVIDIA-side environment (debug/unsloth.nvidia/devenv.lock) so both
  # environments resolve against the same package set. The rev is embedded in
  # the URL (rather than the "nixos-unstable" branch ref) so `nix flake check`
  # reproduces this exact commit instead of re-pinning to whatever unstable is
  # HEAD today.
  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/5b2c2d84341b2afb5647081c1386a80d7a8d8605";
    flake-utils.url = "github:numtide/flake-utils";
  };

  # ROCm has no aarch64-linux or Darwin support, so this shell is x86_64-linux only.
  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachSystem [ "x86_64-linux" ] (system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        lib = pkgs.lib;

        rocm = pkgs.rocmPackages;

        # ROCm runtime shared libraries that the prebuilt (pip-installed) PyTorch /
        # bitsandbytes / triton wheels dlopen() at runtime — libamdhip64.so,
        # librocblas.so, libhipblas.so, etc. On NixOS these are not at the FHS
        # paths the manylinux wheels expect, so we surface them on LD_LIBRARY_PATH.
        # "/run/opengl-driver" is NixOS's driver bind-mount (kernel/userspace GPU
        # driver) — included literally as a string, it is not a Nix package.
        rocmRuntimeLibs = lib.makeLibraryPath [
          rocm.clr
          rocm.rocblas
          rocm.hipblas
          rocm.rocm-smi
          # torch's rocm wheel links libzstd.so.1; surface it so `import torch` resolves.
          pkgs.zstd
          "/run/opengl-driver"
        ];
      in
      {
        devShells.default = pkgs.mkShell {
          # Nix's role here is deliberately minimal: it supplies generic dev
          # tooling plus the ROCm *runtime* libraries only. It must NOT build or
          # replace any AMD-shipped software — torch/bitsandbytes/triton come from
          # AMD's prebuilt wheels (uv pip install --index-url .../whl/rocm7.1),
          # exactly as Unsloth's official AMD docs prescribe. AMD does not certify
          # NixOS, and pkgs.rocmPackages.* is community-maintained, so we keep the
          # closure lean: no hipcc/hip-common/rocsolver/miopen (no HIP compilation
          # happens here — prebuilt wheels only).
          packages = [
            # Generic dev tooling
            pkgs.git
            pkgs.cmake
            pkgs.ninja
            pkgs.ccache
            pkgs.pkg-config
            pkgs.openssl
            pkgs.zlib
            pkgs.zstd
            pkgs.curl
            pkgs.uv
            pkgs.python312
            pkgs.nodejs_22
            pkgs.pre-commit

            # ROCm runtime + introspection tools
            rocm.clr
            rocm.rocblas
            rocm.hipblas
            rocm.rocm-smi
            rocm.rocminfo
          ];

          # ROCM_PATH / HIP_PATH point at the CLR (Common Language Runtime) store
          # path so tooling that reads them finds the HIP runtime root.
          ROCM_PATH = "${rocm.clr}";
          HIP_PATH = "${rocm.clr}";
          LD_LIBRARY_PATH = rocmRuntimeLibs;

          # Pin every HIP process spawned in this shell to a single GPU. This
          # host has 2 visible AMD GPUs, and a native SIGSEGV inside
          # libamdhip64's stream teardown (hip::Device::NullStream() ->
          # HostQueue::terminate() -> ReferenceCountedObject::release()) has
          # been hit — in two independent code paths — whenever both GPUs are
          # visible to a process (the training subprocess, since fixed in
          # worker.py, and a bare `import unsloth` in devenv.nix's enterShell
          # guard). Training never benefits from multi-GPU visibility here
          # anyway (Data Parallel GPUs = 1 even with 2 visible), so we default
          # the whole shell to one GPU instead of patching each call site. This
          # is a default, not a hard lock: a user who wants both GPUs for their
          # own experimentation can `export HIP_VISIBLE_DEVICES=0,1` (or unset
          # it) after entering the shell. See nix/CLAUDE.md failure-mode #7.
          HIP_VISIBLE_DEVICES = "0";

          # HSA_OVERRIDE_GFX_VERSION is intentionally NOT set: it is
          # hardware-specific (it spoofs the gfx arch reported to ROCm for GPUs
          # the shipped libraries don't natively target). The correct value
          # depends on the user's exact GPU, so they must set it themselves if
          # their card needs it. Likewise ROCR_VISIBLE_DEVICES /
          # CUDA_VISIBLE_DEVICES are left to the user/CI.

          # All packages above have non-restrictive licenses (verified for clr,
          # rocblas, hipblas, rocm-smi, rocminfo), so allowUnfree is NOT needed.
          # Re-verify on any nixpkgs bump in case upstream relicenses.

          # Runtime (not eval-time) GPU probe — purely informational. It never
          # gates which packages the shell provides: this flake is
          # unconditionally AMD-only and does no hardware-detection branching.
          shellHook = ''
            # rocmPackages.clr (and friends) ship a Nix setup-hook that exports
            # HIP_DEVICE_LIB_PATH pointing at nixpkgs' rocm-device-libs bitcode
            # (currently the 7.2-era 22.0.0-rocm build). That's for people who
            # COMPILE HIP code against the nix toolchain — we don't: we only
            # dlopen runtime .so's for the pip-installed prebuilt +rocm7.1 torch.
            # A device-libs *bitcode* version skew (7.2 bitcode vs the torch
            # wheel's bundled 7.1) is NOT compatible the way the runtime .so's
            # are, and deterministically SIGSEGVs at HIP device-init. Unlike the
            # setup-hook, this `unset` runs LAST (shellHook is after all setup
            # hooks), so it reliably clears the leak. See DEVENV.md's "why
            # rocm7.1 wheel" note and nix/CLAUDE.md failure-mode #9. Kept in
            # parity with devenv.nix's enterShell.
            unset HIP_DEVICE_LIB_PATH

            if [ -e /dev/kfd ]; then
              echo "AMD KFD device found"
              rocminfo | grep gfx || true
            else
              echo "WARNING: /dev/kfd not found -- no AMD GPU detected on this host"
            fi
          '';
        };
      }
    );
}
