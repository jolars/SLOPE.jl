using Documenter
using SLOPE

is_ci = get(ENV, "CI", nothing) == "true"

makedocs_kwargs = Dict{Symbol, Any}(
    :sitename => "SLOPE",
    :format => Documenter.HTML(
        assets = ["assets/favicon.ico"],
        prettyurls = is_ci
    ),
    :modules => [SLOPE],
    :pages => [
        "Home" => "index.md",
        "Tutorial" => "tutorial.md",
        "API Reference" => "api.md",
    ]
)

# Disable source links locally to avoid NixOS git issues
if !is_ci
    makedocs_kwargs[:remotes] = nothing
end

makedocs(; makedocs_kwargs...)

if is_ci
    deploydocs(
        repo = "github.com/jolars/SLOPE.jl.git",
    )
end
