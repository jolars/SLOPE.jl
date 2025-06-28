using SLOPE
using Test
using SparseArrays
using Random

@testset "Simple design" begin
  n = 3
  p = 2

  x = [1.1 2.3; 0.5 1.5; 0.5 0.2]
  β = [1.0, 2.0]
  y = x * β

  alpha = 1e-12
  lambda = [1.0, 1.0]

  res = slope(x, y, α=alpha, λ=lambda)

  @test res.coefficients[1] ≈ β

  n = 20
  p = 2
  m = 1
end

@testset "Sparse and dense agreement" begin
  Random.seed!(58)

  n = 100
  p = 2

  x = SparseArrays.sprand(n, p, 0.5)
  y = rand(n)

  res_sparse = slope(x, y)
  res_dense = slope(Matrix(x), y)

  @test isapprox(res_dense.coefficients, res_sparse.coefficients, atol=1e-7, rtol=1e-4)
end

