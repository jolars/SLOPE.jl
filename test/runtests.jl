using SLOPE
using Test
using Aqua

@testset "SLOPE" begin
  @testset "Quadratic" begin
    include("quadratic.jl")
  end

  @testset "Logistic" begin
    include("logistic.jl")
  end

  @testset "Multinomial Logistic" begin
    include("multinomial.jl")
  end

  @testset "Aqua" begin
    Aqua.test_all(SLOPE)
  end
end

