using SLOPE
using Test
using SparseArrays
using Random
using LinearAlgebra  # Added for norm function

n = 20
p = 2
m = 3

x = [
  1.2 -0.3;
  -0.5 0.7;
  0.8 -1.2;
  -1.1 0.4;
  0.3 -0.8;
  1.5 0.2;
  -0.2 -0.5;
  0.7 1.1;
  -0.9 -0.9;
  0.4 0.6;
  0.1 -1.0;
  -1.3 0.3;
  0.6 -0.7;
  -0.7 0.8;
  1.1 -0.4;
  -0.4 1.3;
  0.9 -0.6;
  -1.0 0.5;
  0.5 -1.1;
  -0.8 0.9
]

y = [1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2, 3, 1, 2]

# expected_coef = [
#   0.2310035 0.1333503;
#   0.3364695 0.6209708
# ]
# expected_intercept = [0.1775123, 0.1730744]

@testset "Regularization with intercept" begin
  # Set the lambda sequence directly
  α = 0.002
  λ = [4.0, 3.0, 2.0, 1.0]

  # Fit the model with multinomial loss
  @test_nowarn fit = slope(
    x,
    y,
    α=α,
    λ=λ,
    loss=:multinomial,
    fit_intercept=true,
    centering="none",
    scaling="none",
    tol=1e-8,
    max_it=2000
  )

  # coefs = fit.coefficients[end]
  # intercepts = fit.intercepts[end]

end
