{
  description = "A basic flake with a shell";
  inputs.nixpkgs.url = "github:NixOS/nixpkgs/nixos-unstable";
  inputs.nixpkgs-old.url = "github:NixOS/nixpkgs?ref=8a156b259a57d9808441147548613548d7598f4f";
  inputs.systems.url = "github:nix-systems/default";
  inputs.flake-utils = {
    url = "github:numtide/flake-utils";
    inputs.systems.follows = "systems";
  };

  outputs =
    {
      nixpkgs,
      nixpkgs-old,
      flake-utils,
      ...
    }:
    flake-utils.lib.eachDefaultSystem (
      system:
      let
        pkgs = nixpkgs.legacyPackages.${system};
        oldPkgs = nixpkgs-old.legacyPackages.${system};

        buildEnv = pkgs.buildFHSEnv {
          name = "build-env";
          targetPkgs = pkgs: [
            pkgs.bashInteractive
            pkgs.gcc-unwrapped
            pkgs.binutils-unwrapped
            pkgs.glibc
            pkgs.cmake
            pkgs.pkg-config
            pkgs.nodejs
            pkgs.julia-bin
            (pkgs.writeShellScriptBin "julia17" ''
              exec ${oldPkgs.julia_17-bin}/bin/julia "$@"
            '')
          ];
          runScript = "bash";
        };
      in
      {
        devShells.default = buildEnv.env;
      }
    );
}
