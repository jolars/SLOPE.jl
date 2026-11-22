using Random
using SparseArrays

let rng = MersenneTwister(381)
    x = randn(rng, 40, 3)
    y = 1.5 .+ x[:, 1] .- 0.4 .* x[:, 2] .+ 0.2 .* randn(rng, 40)
    weights = range(0.0, 2.0; length = length(y))
    options = (weights = weights, tol = 1.0e-7)

    @testset "Relative and absolute penalties agree" begin
        α_max = only(slope(x, y; options..., path_length = 1).α)
        fractions = [1.5, 1.0, 0.3, 0.0]
        absolute = slope(x, y; options..., α = fractions .* α_max)
        relative = slope(x, y; options..., α = fractions, alpha_scale = :relative)
        sparse_relative = slope(sparse(x), y; options..., α = fractions, alpha_scale = :relative)

        @test relative.α ≈ fractions .* α_max
        @test sparse_relative.α ≈ relative.α
        @test all(iszero, relative.coefficients[2])
        @test predict(relative, x) ≈ predict(absolute, x)
        @test predict(sparse_relative, x) ≈ predict(absolute, x) atol = 1.0e-5
    end

    @testset "Automatic paths and zero penalties" begin
        automatic = slope(x, y; path_length = 5)
        relative = slope(x, y; path_length = 5, alpha_scale = :relative)
        @test relative.α == automatic.α
        @test predict(relative, x) ≈ predict(automatic, x)
        @test slope(x, y; α = 0.5, λ = zeros(size(x, 2)), alpha_scale = :relative).α == [0.0]
    end

    @testset "Validation" begin
        @test_throws ArgumentError slope(x, y; alpha_scale = :fraction)
        for invalid in (-0.1, Inf, NaN, [0.5, 1.0])
            @test_throws ArgumentError slope(x, y; α = invalid, alpha_scale = :relative)
        end
    end
end
