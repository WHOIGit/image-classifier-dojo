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
        ];

        LD_LIBRARY_PATH = "${wheelLibs}:/run/opengl-driver/lib";

        shellHook = ''
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
