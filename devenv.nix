{
  pkgs,
  ...
}:

{
  packages = [
    pkgs.git
    pkgs.bashInteractive
    pkgs.cmake
    pkgs.go-task
  ];

  languages = {
    julia = {
      enable = true;
      package = (
        pkgs.julia-bin.withPackages [
          "Plots"
          "CxxWrap"
          "DataFrames"
          "Distributions"
          "LinearAlgebra"
          "Random"
          "RecipesBase"
          "SparseArrays"
          "StatsBase"
          "slope_jll"
          "Aqua"
          "Test"
          "LanguageServer"
          "Revise"
          "Documenter"
          "RDatasets"
          "Statistics"
        ]
      );
    };
  };
}
