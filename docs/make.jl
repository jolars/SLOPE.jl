using Documenter
using SLOPE

# Copy CHANGELOG.md to docs/src/CHANGELOG.md
cp(
  joinpath(@__DIR__, "../CHANGELOG.md"),
  joinpath(@__DIR__, "src/CHANGELOG.md");
  force=true
)

makedocs(
  sitename="SLOPE",
  format=Documenter.HTML(
    assets=["assets/favicon.ico"],
  ),
  modules=[SLOPE],
  pages=[
    "Home" => "index.md",
    "API Reference" => "api.md",
    "Changelog" => "CHANGELOG.md",
  ]
)

deploydocs(
  repo="https://github.com/jolars/SLOPE.jl.git"
)

