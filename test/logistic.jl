using SLOPE
using Test
using SparseArrays
using Random

n = 10
p = 3

x = [
    0.288 -0.0452 0.88;
    0.788 0.576 -0.305;
    1.51 0.39 -0.621;
    -2.21 -1.12 -0.0449;
    -0.0162 0.944 0.821;
    0.594 0.919 0.782;
    0.0746 -1.99 0.62;
    -0.0561 -0.156 -1.47;
    -0.478 0.418 1.36;
    -0.103 0.388 -0.0538
]

# Fixed coefficients beta
beta = [0.5, -0.1, 0.2]

# Compute linear predictor
linear_predictor = x * beta

# Compute probabilities using logistic function
prob = 1.0 ./ (1.0 .+ exp.(-linear_predictor))

# Generate deterministic response variable y
y = Float64.(prob .> 0.5)

α = 0.05
lambda = [2.128045, 1.833915, 1.644854]

@testset "No intercept, no standardization" begin
    coef_target = [1.3808558, 0.0, 0.3205496]

    fit = slope(
        x, y;
        λ = lambda,
        loss = :logistic,
        fit_intercept = false,
        centering = :none,
        scaling = :none,
        α = α,
        tol = 1.0e-7
    )

    coefs = fit.coefficients[1]

    @test isapprox(coefs, coef_target, atol = 1.0e-6)
end

@testset "Intercept, no standardization" begin
    coef_target = [1.2748806, 0.0, 0.2062611]
    intercept_target = [0.3184528]

    fit = slope(
        x, y;
        λ = lambda,
        loss = :logistic,
        fit_intercept = true,
        centering = :none,
        scaling = :none,
        α = α,
        tol = 1.0e-7
    )

    coefs = fit.coefficients[1]
    intercept = fit.intercepts[1]

    @test isapprox(intercept, intercept_target, atol = 1.0e-4)
end

@testset "Predictions" begin
    Random.seed!(5)

    n = 50
    p = 3

    x = rand(n, p)
    β = [0.6, 0.0, -0.9]

    eta = x * β
    prob = 1.0 ./ (1.0 .+ exp.(-eta))
    y = Float64.(prob .> 0.5)

    res = slope(x, y, loss = :logistic)

    predictions = predict(res, x)

    @test length(predictions[1]) == n
    @test unique(vcat([vec(pred) for pred in predictions]...)) ⊆ [0.0, 1.0]
end
