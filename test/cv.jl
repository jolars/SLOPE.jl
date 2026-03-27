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
    @test 1 <= res.best_ind <= length(res.results)
    @test 1 <= res.best_α_ind <= length(res.results[res.best_ind].alphas)
    @test isapprox(best_α(res), res.results[res.best_ind].alphas[res.best_α_ind])
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

@testset "Best model extraction" begin
    Random.seed!(22)

    n = 80
    p = 6

    x = randn(n, p)
    β = [1.2, -0.8, 0.0, 0.0, 0.3, 0.0]
    y = x * β + 0.3 * randn(n)

    res = slopecv(x, y, q = [0.1, 0.2], γ = [0.0], n_folds = 4)
    fit = best_model(res)

    @test fit isa SlopeFit
    @test length(fit.α) == 1
    @test isapprox(fit.α[1], best_α(res))
    @test refit(res, x = x, y = y) isa SlopeFit
end

@testset "Best model delegates to refit for γ ≠ 0" begin
    Random.seed!(23)

    n = 80
    p = 6
    x = randn(n, p)
    y = randn(n)

    res = slopecv(x, y, q = [0.1], γ = [1.0], n_folds = 4)

    @test_throws ArgumentError best_model(res)
    @test_throws ArgumentError best_model(res, λ = regweights(p))
    @test best_model(res, x = x, y = y, λ = regweights(p)) isa SlopeFit
end

@testset "Refit from CV result" begin
    Random.seed!(24)

    n = 60
    p = 8
    x = randn(n, p)
    y = x[:, 1] .- 0.5 .* x[:, 2] .+ 0.2 * randn(n)

    res = slopecv(x, y, q = [0.05, 0.1], γ = [0.0], n_folds = 3)
    fit = refit(res, x = x, y = y)

    @test fit isa SlopeFit
    @test length(fit.α) == 1
    @test isapprox(fit.α[1], best_α(res))
end

@testset "Refit requires explicit x/y" begin
    Random.seed!(25)

    n = 40
    p = 5
    x = randn(n, p)
    y = randn(n)
    res = slopecv(x, y, q = [0.1], γ = [0.0], n_folds = 3)

    @test_throws ArgumentError refit(res)
    @test_throws ArgumentError refit(res, x = x)
    @test_throws ArgumentError refit(res, y = y)
end

@testset "Refit validates measure" begin
    Random.seed!(26)

    x = randn(50, 6)
    y = randn(50)
    res = slopecv(x, y, q = [0.1], γ = [0.0], metric = :mse, n_folds = 3)

    @test_throws ArgumentError refit(res, x = x, y = y, measure = :mae)
end

@testset "Refit behavior for nonzero γ" begin
    Random.seed!(27)

    n = 50
    p = 6
    x = randn(n, p)
    y = randn(n)
    λ_override = regweights(p)

    res = slopecv(x, y, q = [0.1], γ = [1.0], n_folds = 3)
    # Nonzero γ requires explicit λ for refit.
    @test_throws ArgumentError refit(res)

    fit = refit(res, x = x, y = y, λ = λ_override)
    @test fit isa SlopeFit
    @test length(fit.α) == 1

    # User can still override data when passing explicit λ.
    fit_override = refit(res, x = x, y = y, λ = λ_override)
    @test fit_override isa SlopeFit
end
