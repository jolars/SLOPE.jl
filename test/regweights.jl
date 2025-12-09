using SLOPE
using Test

@testset "BH Sequence" begin
    output = regweights(4, q = 0.1, type = :bh)
    expected = [2.241403, 1.959964, 1.780464, 1.644854]

    @test isapprox(output, expected, atol = 1.0e-6)

    # Check assertions
    @test_throws DomainError regweights(0, q = 0.1, type = :bh)
    @test_throws DomainError regweights(4, q = -0.1, type = :bh)
end

@testset "Gaussian Sequence" begin
    output = regweights(4, q = 0.4, type = :gaussian, n = 100)
    expected = [1.6448536, 1.2991214, 1.0596442, 0.8654631]

    @test isapprox(output, expected, atol = 1.0e-3)

    # Check assertions
    @test_throws DomainError regweights(4, q = 1.5, type = :gaussian, n = 100)
    @test_throws ArgumentError regweights(4, q = 1.5, type = :gaussian)
end

@testset "Oscar Sequence" begin
    output = regweights(3, type = :oscar, θ1 = 5, θ2 = 1)
    expected = [7, 6, 5]

    @test isequal(output, expected)

    # Check assertions
    @test_throws DomainError regweights(3, type = :oscar, θ1 = -1, θ2 = 1)
    @test_throws DomainError regweights(3, type = :oscar, θ1 = 1, θ2 = -1)
end

@testset "Lasso Sequence" begin
    output = regweights(5, type = :lasso)
    expected = [1, 1, 1, 1, 1]

    @test isequal(output, expected)
end

@testset "Invalid Type" begin
    @test_throws ArgumentError regweights(4, type = :invalid_type)
end
