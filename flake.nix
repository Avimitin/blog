{
  description = "Generic devshell setup";

  inputs = {
    # The nixpkgs
    nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";

    # Utility functions
    flake-utils.url = "github:numtide/flake-utils";
  };

  outputs = { self, nixpkgs, flake-utils }:
    let
      pkgsForSys = system: import nixpkgs { inherit system; };
      perSystem = (system:
        let
          pkgs = pkgsForSys system;
        in
        {
          devShells.default = pkgs.mkShell {
            buildInputs = [
              pkgs.hugo
            ];
          };

          formatter = pkgs.nixpkgs-fmt;
          legacyPackages = pkgs;
        });
    in
    {
      # Other system-independent attr
    } //

    flake-utils.lib.eachDefaultSystem perSystem;
}
