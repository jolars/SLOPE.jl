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

  @test res.β[1] ≈ β

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

  path_length = 3

  res_sparse = slope(x, y)
  res_dense = slope(Matrix(x), y)

  @test maximum(abs.(maximum.(res_dense.β - res_sparse.β))) < 1e-10
end

