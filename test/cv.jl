using SLOPE
using Test
using SparseArrays
using Random

@testset "Sparse and dense agreement" begin
    Random.seed!(14)

    n = 1000
    p = 20

    x = SparseArrays.sprand(n, p, 0.5)
    y = rand(n)

    q = [0.1, 0.2]

    Random.seed!(8)
    res_sparse = slopecv(x, y, q = q)

    Random.seed!(8)
    res_dense = slopecv(Matrix(x), y, q = q)

    @test isapprox(res_dense.best_score, res_sparse.best_score)
end

@testset "Multiple parameters" begin
    Random.seed!(10)

    n = 1000
    p = 20

    x = SparseArrays.sprand(n, p, 0.5)
    y = rand(n)

    q = [0.1, 0.2]
    γ = [0.0, 0.5, 1.0]

    res = slopecv(x, y, q = q, γ = γ)

    @test length(res.results) == length(q) * length(γ)
end

@testset "Plots" begin
    Random.seed!(13)

    n = 100
    p = 2

    x = rand(n, p)
    y = rand(n)

    q = [0.1, 0.2]
    γ = [0.0, 1.0]

    res = slopecv(x, y, q = q, γ = γ)

    plt = plot(res)

    @test plt isa Plots.Plot

end
