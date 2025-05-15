using SLOPE
using Test
using SparseArrays

@testset "Basic" begin
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
