using SLOPE
using Test
using Plots
using Random

@testset "Univariate Family Plot" begin
  n = 3
  p = 2

  x = [1.1 2.3; 0.5 1.5; 0.5 0.2]
  β = [1.0, 2.0]
  y = x * β

  lambda = [1.0, 1.0]

  res = slope(x, y)

  plt = plot(res)

  @test !isnothing(plt)
end

@testset "Multinomial Family Plot" begin
  Random.seed!(1)

  n = 100
  p = 3
  m = 4

  x = rand(n, p)

  β = rand(p, m)
  η = x * β + randn(n, m)
  Y = exp.(η) ./ sum(exp.(η), dims=2)
  y = Vector{Int}(undef, n)

  for i in 1:n
    y[i] = argmax(Y[i, :])
  end

  res = slope(x, y, loss=:multinomial)

  plt = plot(res)

  @test plt isa Plots.Plot
end
