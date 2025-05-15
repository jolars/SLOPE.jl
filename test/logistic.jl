using SLOPE
using Test
using SparseArrays
using Random

n = 10
p = 3

x = [
  0.288 -0.0452 0.880;
  0.788 0.576 -0.305;
  1.510 0.390 -0.621;
  -2.210 -1.120 -0.0449;
  -0.0162 0.944 0.821;
  0.594 0.919 0.782;
  0.0746 -1.990 0.620;
  -0.0561 -0.156 -1.470;
  -0.478 0.418 1.360;
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
  coef_target = [1.3808558, 0.0000000, 0.3205496]

  fit = slope(x, y;
    λ=lambda,
    loss="logistic",
    fit_intercept=false,
    centering="none",
    scaling="none",
    α=α,
    tol=1e-7)

  coefs = fit.β[1]

  @test isapprox(coefs, coef_target, atol=1e-6)
end

@testset "Intercept, no standardization" begin
  coef_target = [1.2748806, 0.0, 0.2062611]
  intercept_target = 0.3184528

  fit = slope(x, y;
    λ=lambda,
    loss="logistic",
    fit_intercept=true,
    centering="none",
    scaling="none",
    α=α,
    tol=1e-7)

  coefs = fit.β[1]
  intercept = fit.β0[1]

  @test isapprox(intercept, intercept_target, atol=1e-4)
end
