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
            # `uvx pyright` cannot run here: pyright-python falls back to a
            # generic-linux node binary that NixOS will not exec. This build
            # wraps its own node. Keep the CI pins in .github/workflows/
            # test.yml matching the versions this nixpkgs provides.
            pkgs.pyright
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
