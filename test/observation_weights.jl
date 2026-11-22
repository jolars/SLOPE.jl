using Random
using SparseArrays

let x = [-1.2 0.0; -0.8 0.2; -0.5 -0.7; -0.2 0.0;
         0.0 1.3; 0.2 -0.5; 0.4 0.8; 0.6 0.0;
         0.8 -0.2; 1.0 0.9; 1.2 -0.4; 1.4 0.3]
    y = 0.3 .+ x[:, 1] .- 0.4 .* x[:, 2]
    y[5] += 3.0
    weights = [1, 2, 1, 3, 0, 1, 2, 1, 1, 2, 1, 1]

    @testset "Fit" begin
        rows = [i for i in eachindex(weights) for _ in 1:weights[i]]
        options = (α = 0.05, tol = 1.0e-7)
        weighted = slope(x, y; options..., weights = view(weights, :))
        repeated = slope(x[rows, :], y[rows]; options...)
        sparse_weighted = slope(sparse(x), y; options..., weights)
        uniform = slope(x, y; options..., weights = ones(length(y)))
        unweighted = slope(x, y; options...)

        @test only(predict(weighted, x)) ≈ only(predict(repeated, x)) atol = 1.0e-5
        @test only(predict(weighted, x)) ≈ only(predict(sparse_weighted, x)) atol = 1.0e-5
        @test predict(uniform, x) ≈ predict(unweighted, x)
        @test !isapprox(predict(weighted, x), predict(unweighted, x))
    end

    @testset "Cross-validation and refitting" begin
        options = (n_folds = 3, path_length = 8)
        input_weights = copy(weights)
        Random.seed!(42)
        cv = slopecv(x, y; options..., weights = input_weights)
        Random.seed!(42)
        sparse_cv = slopecv(sparse(x), y; options..., weights)
        Random.seed!(42)
        unweighted_cv = slopecv(x, y; options...)
        Random.seed!(42)
        uniform_cv = slopecv(x, y; options..., weights = ones(length(y)))

        scores = only(cv.results).scores
        @test scores ≈ only(sparse_cv.results).scores atol = 1.0e-5
        @test only(uniform_cv.results).scores ≈ only(unweighted_cv.results).scores
        @test !isapprox(scores[:, 1], only(unweighted_cv.results).scores[:, 1])

        selected = (α = best_α(cv), q = cv.best_params["q"])
        expected = slope(x, y; selected..., weights)
        fill!(input_weights, 1)
        @test predict(best_model(cv), x) ≈ predict(expected, x)
        @test predict(refit(cv; x, y), x) ≈ predict(expected, x)

        rows = 1:8
        x_refit = x[rows, :]
        y_refit = y[rows]
        expected = slope(x_refit, y_refit; selected..., weights = weights[rows])
        refitted = best_model(cv; x = x_refit, y = y_refit, weights = weights[rows])
        @test predict(refitted, x) ≈ predict(expected, x)

        expected = slope(x, y; selected...)
        @test predict(refit(cv; x, y, weights = nothing), x) ≈ predict(expected, x)

        @test_throws ArgumentError refit(cv; x, y, alpha_scale = :relative)
        expected = slope(x, y; α = 0.5, alpha_scale = :relative, weights)
        refitted = refit(cv; x, y, α = 0.5, alpha_scale = :relative)
        @test predict(refitted, x) ≈ predict(expected, x)
    end

    @testset "Validation" begin
        for fit in (slope, slopecv)
            @test_throws DimensionMismatch fit(x, y; weights = weights[2:end])
            @test_throws DimensionMismatch fit(x, y; weights = Float64[])
            for invalid in (zeros(length(y)), fill(-1, length(y)), fill(NaN, length(y)), fill(Inf, length(y)))
                @test_throws ArgumentError fit(x, y; weights = invalid)
            end
            @test_throws ArgumentError fit(x, y; weights, λ = :gaussian)
        end

        one_positive = zeros(length(y))
        one_positive[1] = 1.0
        @test_throws ErrorException slopecv(x, y; n_folds = 3, weights = one_positive)
    end
end
