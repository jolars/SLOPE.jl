using SLOPE
using Test
using SparseArrays
using Random

@testset "coef function" begin
    @testset "Single response model (m=1)" begin
        Random.seed!(123)
        n, p = 50, 10
        x = randn(n, p)
        y = x[:, 1:3] * ones(3) + randn(n)
        fit = slope(x, y)

        # Test default behavior: all coefficients as sparse matrices
        all_coefs = coef(fit)
        @test isa(all_coefs, Vector)
        @test length(all_coefs) > 0
        @test isa(all_coefs[1], SparseArrays.SparseMatrixCSC)
        @test size(all_coefs[1]) == (p, 1)

        # Test specific index
        coef_1 = coef(fit, index = 1)
        @test isa(coef_1, SparseArrays.SparseMatrixCSC)
        @test size(coef_1) == (p, 1)
        @test coef_1 == all_coefs[1]

        # Test simplify=true for all coefficients (should return 2D matrix)
        coef_matrix = coef(fit, simplify = true)
        @test isa(coef_matrix, Array)
        @test size(coef_matrix, 1) == p
        @test size(coef_matrix, 2) == length(all_coefs)
        @test coef_matrix[:, 1] ≈ vec(all_coefs[1])

        # Test simplify=true with specific index (should return vector)
        coef_vec = coef(fit, index = 1, simplify = true)
        @test length(coef_vec) == p
        @test coef_vec ≈ vec(all_coefs[1])
    end

    @testset "Multinomial model (m>1)" begin
        Random.seed!(456)
        n, p = 150, 10
        x = randn(n, p)
        y = repeat([1, 2, 3], 50)
        fit = slope(x, y, loss = :multinomial)

        # Test default behavior
        all_coefs = coef(fit)
        @test isa(all_coefs, Vector)
        @test length(all_coefs) > 0
        @test isa(all_coefs[1], SparseArrays.SparseMatrixCSC)
        p_size, m_size = size(all_coefs[1])
        @test p_size == p
        @test m_size > 1  # Multinomial has multiple responses

        # Test specific index
        coef_idx = coef(fit, index = 1)
        @test isa(coef_idx, SparseArrays.SparseMatrixCSC)
        @test size(coef_idx) == (p_size, m_size)
        @test coef_idx == all_coefs[1]

        # Test simplify=true for all coefficients (should return 3D array)
        coef_array = coef(fit, simplify = true)
        @test isa(coef_array, Array)
        @test ndims(coef_array) == 3
        @test size(coef_array, 1) == p_size
        @test size(coef_array, 2) == m_size
        @test size(coef_array, 3) == length(all_coefs)

        # Test simplify=true with specific index (should return matrix, not vector)
        coef_mat = coef(fit, index = 1, simplify = true)
        @test isa(coef_mat, SparseArrays.SparseMatrixCSC)
        @test size(coef_mat) == (p_size, m_size)
    end

    @testset "Error handling" begin
        Random.seed!(789)
        n, p = 30, 5
        x = randn(n, p)
        y = randn(n)
        fit = slope(x, y)
        path_length = length(fit.coefficients)

        # Test out of bounds index
        @test_throws BoundsError coef(fit, index = 0)
        @test_throws BoundsError coef(fit, index = path_length + 1)
        @test_throws BoundsError coef(fit, index = -1)
    end
end
