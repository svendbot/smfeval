{
  description = "smfeval: probabilistic SLAM evaluation";

  inputs = {
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs { inherit system; };
        python = pkgs.python312;
        pythonEnv = python.withPackages (ps: with ps; [
          pip
          hatchling
          hatch
          twine
          build
          jupytext
        ]);
      in
      {
        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.uv
            pkgs.ruff
            # For `make typecheck`. pyright-python runs the global node when
            # it finds one and otherwise downloads a generic-linux binary
            # that NixOS cannot exec, so without this `uvx pyright` fails
            # here. Supplying node rather than pkgs.pyright keeps local and
            # CI on the one pinned version (nixpkgs lags it).
            pkgs.nodejs
          ];

          shellHook = ''
            export PYTHONPATH="$PWD:$PYTHONPATH"
            export UV_PYTHON=${pkgs.python312}/bin/python
            export UV_PYTHON_DOWNLOADS=never
            export LD_LIBRARY_PATH="${pkgs.lib.makeLibraryPath [ pkgs.stdenv.cc.cc.lib pkgs.zlib ]}:$LD_LIBRARY_PATH"
            echo "smfeval dev shell, $(python --version)"
          '';
        };
      });
}
