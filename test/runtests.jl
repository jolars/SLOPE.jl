using SLOPE
using Test
using SparseArrays

@testset "Simple design" begin
  n = 3
  p = 2

  x = [1.1 2.3; 0.5 1.5; 0.5 0.2]
  β = [1.0, 2.0]
  y = x * β

  alpha = Float64[1e-12]
  lambda = [1.0, 1.0]

  res = slope(x, y, α=alpha, λ=lambda)

  @test res.β[1] ≈ β
end

@testset "Dense sparse agreement" begin
  n = 20
  p = 2
  m = 1

  x = SparseArrays.sprand(n, p, 0.5)
  y = rand(n)

  path_length = 3

  alpha = Float64[]
  lambda = Float64[]

  coef_vals = Float64[]
  coef_rows = Int[]
  coef_cols = Int[]
  nnz = Int[]

  res_sparse = slope(x, y, α=alpha, λ=lambda)
  res_dense = slope(Matrix(x), y, α=alpha, λ=lambda)

  @test maximum(abs.(maximum.(res_dense.β - res_sparse.β))) < 1e-10
end

include(joinpath(@__DIR__, "aqua.jl"))
