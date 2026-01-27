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

    @testset "Alpha parameter" begin
        Random.seed!(321)
        n, p = 50, 10
        x = randn(n, p)
        y = x[:, 1:3] * ones(3) + randn(n)
        fit = slope(x, y)

        alphas = fit.α

        # Test exact alpha match
        alpha_mid = alphas[div(length(alphas), 2)]
        coef_exact = coef(fit, α = alpha_mid)
        coef_idx = coef(fit, index = div(length(alphas), 2))
        @test coef_exact ≈ Matrix(coef_idx)

        # Test interpolation between two alphas
        idx1 = 10
        idx2 = 11
        alpha_between = (alphas[idx1] + alphas[idx2]) / 2
        coef_interp = coef(fit, α = alpha_between)

        # Verify it's between the two coefficient sets
        coef1 = Matrix(fit.coefficients[idx1])
        coef2 = Matrix(fit.coefficients[idx2])
        @test size(coef_interp) == size(coef1)

        # Check interpolation is reasonable (should be between the two)
        for i in 1:length(coef_interp)
            if coef1[i] != coef2[i]
                min_val = min(coef1[i], coef2[i])
                max_val = max(coef1[i], coef2[i])
                @test min_val <= coef_interp[i] <= max_val || 
                      isapprox(coef_interp[i], min_val, atol=1e-10) || 
                      isapprox(coef_interp[i], max_val, atol=1e-10)
            end
        end

        # Test with simplify=true
        coef_vec = coef(fit, α = alpha_between, simplify = true)
        @test isa(coef_vec, Vector)
        @test length(coef_vec) == p

        # Test boundary alphas
        coef_min = coef(fit, α = minimum(alphas))
        coef_max = coef(fit, α = maximum(alphas))
        @test size(coef_min) == (p, 1)
        @test size(coef_max) == (p, 1)
    end

    @testset "Error handling" begin
        Random.seed!(789)
        n, p = 30, 5
        x = randn(n, p)
        y = randn(n)
        fit = slope(x, y)
        path_length = length(fit.coefficients)
        alphas = fit.α

        # Test out of bounds index
        @test_throws BoundsError coef(fit, index = 0)
        @test_throws BoundsError coef(fit, index = path_length + 1)
        @test_throws BoundsError coef(fit, index = -1)

        # Test invalid alpha combinations
        @test_throws ArgumentError coef(fit, index = 1, α = 0.5)
        @test_throws ArgumentError coef(fit, α = 0.5, refit = true)

        # Test alpha out of range
        @test_throws ArgumentError coef(fit, α = minimum(alphas) - 1)
        @test_throws ArgumentError coef(fit, α = maximum(alphas) + 1)
    end
end
