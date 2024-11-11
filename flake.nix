{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/nixos-unstable";
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils, ... }:
    flake-utils.lib.eachDefaultSystem (system:
      let
        pkgs = import nixpkgs {
          inherit system;
        };
      in rec {
        devShell = pkgs.mkShell {
          buildInputs = with pkgs; [
            (python3.withPackages (ps: with ps; [
              numpy
              matplotlib
              scikit-learn
              jupyter
              pandas
              seaborn
              scipy
              nbdime
            ]))
            git
          ];

          #shellHook = ''
          #  echo "Starting Jupyter Notebook..."
          #  jupyter notebook
          #'';
        };
      }
    );
} 
