{
  description = "image-classifier-dojo dev shell";

  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixpkgs-unstable";

  outputs = { self, nixpkgs }:
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
        '';
      };
    };
}
