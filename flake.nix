{
  inputs = {
    nixpkgs.url = "github:nixos/nixpkgs/23.11";
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
            (python310.withPackages (ps: with ps; [
              numpy
              matplotlib
              scikit-learn
              jupyter
              pandas
              seaborn
              scipy
              nbdime
              xlrd
              openpyxl
              keras
              tensorflow
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
