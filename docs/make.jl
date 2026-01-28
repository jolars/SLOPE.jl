using Documenter
using SLOPE

is_ci = get(ENV, "CI", nothing) == "true"

makedocs(
    sitename = "SLOPE",
    format = Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = is_ci
    ),
    modules = [SLOPE],
    pages = [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API Reference" => "api.md",
    ],
    remotes = is_ci ? "https://github.com/jolars/SLOPE.jl" : nothing,
)

if is_ci
    deploydocs(
        repo = "github.com/jolars/SLOPE.jl.git",
    )
end
