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
        # Versions the CI pins must track. CI has no nix in its lint and
        # typecheck jobs, so the `pins` job in .github/workflows/test.yml
        # reads these and fails when a pin drifts from the flake -- e.g.
        # after a `nix flake update`.
        toolVersions = {
          ruff = pkgs.ruff.version;
          pyright = pkgs.pyright.version;
        };

        devShells.default = pkgs.mkShell {
          packages = [
            pythonEnv
            pkgs.uv
            pkgs.ruff
            # `uvx pyright` cannot run here: pyright-python falls back to a
            # generic-linux node binary that NixOS will not exec. This build
            # wraps its own node. Its version is exported as toolVersions
            # below, which CI checks its pin against.
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
