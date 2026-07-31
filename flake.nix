{
  description = "image-classifier-dojo dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { nixpkgs, ... }:
    let
      system = "x86_64-linux";
      pkgs = nixpkgs.legacyPackages.${system};
      python = pkgs.python314;

      wheelLibs = pkgs.lib.makeLibraryPath [
        pkgs.stdenv.cc.cc.lib
        pkgs.zlib
        pkgs.glib
        pkgs.libGL
      ];
    in
    {
      devShells.${system}.default = pkgs.mkShell {
        name = "dojo-shell";
        packages = [
          python
          pkgs.git
          pkgs.git-lfs
        ];

        shellHook = ''
          # Assemble LD_LIBRARY_PATH: Nix-built wheel deps + host GPU driver
          # libs. The driver dir varies by host, so detect it at shell entry
          # (first match wins). Override with DOJO_DRIVER_LIBS if detection
          # picks wrong. Only pure driver-lib dirs are added — never a full
          # system lib dir like /usr/lib/x86_64-linux-gnu, whose glibc would
          # break Nix binaries.
          _driver_libs=""
          if [ -n "''${DOJO_DRIVER_LIBS:-}" ]; then
            _driver_libs="$DOJO_DRIVER_LIBS"
          elif [ -d /run/opengl-driver/lib ]; then
            # NixOS
            _driver_libs="/run/opengl-driver/lib"
          elif [ -d /.singularity.d/libs ]; then
            # Apptainer/Singularity with --nv: driver libs are bound here
            _driver_libs="/.singularity.d/libs"
          elif [ -d /usr/lib/wsl/lib ]; then
            # WSL2: pure driver-lib dir (libcuda, libnvidia-ml, libdxcore)
            _driver_libs="/usr/lib/wsl/lib"
          else
            # Generic FHS distro: symlink just the driver libs into a stub
            # dir so we don't pull in the rest of the system lib dir.
            # Debian/Ubuntu, Fedora/RHEL, Arch paths in that order.
            for _d in /usr/lib/x86_64-linux-gnu /usr/lib64 /usr/lib; do
              if [ -f "$_d/libcuda.so.1" ]; then
                _driver_libs="$HOME/.cache/dojo-cuda-stubs"
                mkdir -p "$_driver_libs"
                ln -sf "$_d"/libcuda.so* "$_driver_libs/" 2>/dev/null || true
                ln -sf "$_d"/libnvidia-ml.so* "$_driver_libs/" 2>/dev/null || true
                break
              fi
            done
            unset _d
          fi
          export LD_LIBRARY_PATH="${wheelLibs}''${_driver_libs:+:$_driver_libs}"
          unset _driver_libs

          if [ ! -d .venv ]; then
            echo "Creating .venv with ${python.name} and installing .[all,dev] ..."
            ${python.interpreter} -m venv .venv
            .venv/bin/pip install --upgrade pip
            .venv/bin/pip install -e ".[all,dev]"
          fi
          source .venv/bin/activate
          echo "dojo dev shell: $(python --version) — venv at .venv"

          # On systems with LDAP/SSSD, getpwuid() fails in the Nix env because
          # the Nix-built glibc can't load the system's libnss_sss.so. Patch the
          # prompt and whoami to use the inherited $USER instead.
          if [ -n "$USER" ] && ! id -un &>/dev/null; then
            PS1="''${PS1//\\u/$USER}"
            export PS1
            whoami() { echo "$USER"; }
          fi

          # On non-NixOS (e.g. Ubuntu), /run/opengl-driver doesn't exist.
          # Add only the CUDA driver libs via a stub dir to avoid contaminating
          # LD_LIBRARY_PATH with the system glibc (which would break Nix binaries).
          if [ ! -d /run/opengl-driver ] && [ -f /usr/lib/x86_64-linux-gnu/libcuda.so.1 ]; then
            _cuda_stubs="$HOME/.cache/dojo-cuda-stubs"
            mkdir -p "$_cuda_stubs"
            ln -sf /usr/lib/x86_64-linux-gnu/libcuda.so* "$_cuda_stubs/" 2>/dev/null || true
            ln -sf /usr/lib/x86_64-linux-gnu/libnvidia-ml.so* "$_cuda_stubs/" 2>/dev/null || true
            export LD_LIBRARY_PATH="$LD_LIBRARY_PATH:$_cuda_stubs"
            unset _cuda_stubs
          fi
        '';
      };
    };
}
