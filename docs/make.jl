using Documenter
using SLOPE

makedocs(
  sitename="SLOPE",
  format=Documenter.HTML(
    assets=["assets/favicon.ico"],
  ),
  modules=[SLOPE],
  pages=[
    "Home" => "index.md",
    "API Reference" => "api.md",
  ]
)

deploydocs(
  repo="https://github.com/jolars/SLOPE.jl.git"
)

