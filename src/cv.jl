using CxxWrap
using Random

struct SlopeCvParameters
    γ::Union{Float64, AbstractVector}
    q::Union{Float64, AbstractVector}
    n_folds::Int
    n_repeats::Int
    metric::Symbol
    fold_indices::Vector{Int}
end

"""
    SlopeGridResult

Results for a specific hyperparameter combination in the SLOPE cross-validation grid search.

# Fields
- `params::Dict{String,Any}`: Dictionary of hyperparameter values (e.g., "q", "γ")
- `scores::Matrix{Real}`: Cross-validation scores for each fold and alpha value
- `alphas::Vector{Real}`: Sequence of alpha values for the regularization path
- `scores_means::Vector{Real}`: Mean score across folds for each alpha
- `scores_errors::Vector{Real}`: Standard errors of scores across folds
"""
struct SlopeGridResult
    params::Dict{String, Any}
    scores::Matrix{Real}
    alphas::Vector{Real}
    scores_means::Vector{Real}
    scores_errors::Vector{Real}
end

"""
    SlopeCvResult

Result structure from SLOPE cross-validation.

# Fields

- `metric::Symbol`: The evaluation metric used (e.g., `:mse`, `:accuracy`)
- `best_score::Real`: The best score achieved during cross-validation
- `best_ind::Int`: Index of the best parameter combination
- `best_α_ind::Int`: Index of the best alpha value in the regularization path
- `best_params::Dict{String,Any}`: Dictionary with the best parameter values
- `results::Vector{SlopeGridResult}`: Grid search results, of type [`SlopeGridResult`](@ref)
  for each parameter combination
- `best_fit::Union{SlopeFit,Nothing}`: Final model fitted at the best CV setting on the
  full dataset when supported (currently when selected `γ ≈ 0`)

"""
struct SlopeCvResult
    metric::Symbol
    best_score::Real
    best_ind::Int
    best_α_ind::Int
    best_params::Dict{String, Any}
    results::Vector{SlopeGridResult}
    best_fit::Union{SlopeFit, Nothing}
    λ_input::Union{AbstractVector, Symbol, Nothing}
    slope_kwargs::Dict{Symbol, Any}
end

function Base.show(io::IO, ::MIME"text/plain", cv::SlopeCvResult)
    n_params = length(cv.results)
    best_result = cv.results[cv.best_ind]
    n_alphas = length(best_result.alphas)

    println(io, "SLOPE cross-validation results")
    println(io)
    println(io, "Metric: ", cv.metric)
    println(io, "Best score: ", round(cv.best_score, sigdigits = 4))
    println(io, "Best α: ", round(best_α(cv), sigdigits = 4))
    println(io)
    println(io, "Best parameters:")
    for (key, val) in sort(collect(cv.best_params))
        if val isa AbstractFloat
            println(io, "  ", key, ": ", round(val, sigdigits = 4))
        else
            println(io, "  ", key, ": ", val)
        end
    end
    println(io)
    println(io, "Grid search:")
    println(io, "  Parameter combinations: ", n_params)
    println(io, "  Alpha values per combination: ", n_alphas)
    println(io, "  Best model available: ", isnothing(cv.best_fit) ? "No" : "Yes")

    # Show summary of all parameter combinations if more than 1
    return if n_params > 1
        println(io)
        println(io, "Parameter combination scores:")
        for (i, result) in enumerate(cv.results)
            best_score_for_combo = minimum(result.scores_means)
            marker = i == cv.best_ind ? " *" : ""
            print(io, "  ", i, ". ")
            param_str = join(["$(k)=$(round(v, sigdigits = 3))" for (k, v) in sort(collect(result.params))], ", ")
            if i < n_params
                println(io, param_str, ": ", round(best_score_for_combo, sigdigits = 4), marker)
            else
                print(io, param_str, ": ", round(best_score_for_combo, sigdigits = 4), marker)
            end
        end
    end
end

# Compact display
function Base.show(io::IO, cv::SlopeCvResult)
    print(io, "SlopeCvResult(")
    print(io, "metric=", cv.metric, ", ")
    print(io, "best_score=", round(cv.best_score, sigdigits = 4), ", ")
    print(io, "best_α=", round(best_α(cv), sigdigits = 4), ", ")
    print(io, "n_combinations=", length(cv.results))
    return print(io, ")")
end

"""
    best_α(cv::SlopeCvResult)

Return the α value selected by cross-validation.
"""
function best_α(cv::SlopeCvResult)
    return cv.results[cv.best_ind].alphas[cv.best_α_ind]
end

"""
    best_model(cv::SlopeCvResult; kwargs...)

Return the final model fitted at the cross-validated best parameters.

For `γ == 0`, this returns the cached refit model built from the `slopecv` inputs.
For other settings, this delegates to [`refit`](@ref), which requires explicit `x` and `y`.
"""
function best_model(cv::SlopeCvResult; kwargs...)
    if isempty(kwargs) && !isnothing(cv.best_fit)
        return cv.best_fit
    end
    if isempty(kwargs)
        throw(ArgumentError("`best_model` requires explicit `x` and `y` when no cached model is available."))
    end
    return refit(cv; kwargs...)
end

function _build_refit_kwargs(
        best_q::Real,
        best_α::Real,
        λ_input::Union{AbstractVector, Symbol, Nothing},
        slope_kwargs::Dict{Symbol, Any},
    )
    refit_kwargs = copy(slope_kwargs)
    refit_kwargs[:α] = best_α

    if λ_input isa AbstractVector
        refit_kwargs[:λ] = λ_input
    else
        refit_kwargs[:λ] = isnothing(λ_input) ? :bh : λ_input
        refit_kwargs[:q] = best_q
    end
    return refit_kwargs
end

function _fit_best_model(
        x::Union{AbstractMatrix, SparseMatrixCSC},
        y::AbstractVector,
        λ_input::Union{AbstractVector, Symbol, Nothing},
        best_q::Real,
        best_γ::Real,
        best_α::Real,
        slope_kwargs::Dict{Symbol, Any},
    )
    if !isapprox(best_γ, 0.0) && !(λ_input isa AbstractVector)
        return nothing
    end

    refit_kwargs = _build_refit_kwargs(best_q, best_α, λ_input, slope_kwargs)
    return slope(x, y; (pairs(refit_kwargs))...)
end

"""
    refit(cv::SlopeCvResult; x, y, measure=nothing, kwargs...)

Refit a SLOPE model using the optimal parameters selected by cross-validation.

# Arguments
- `cv::SlopeCvResult`: Cross-validation result from [`slopecv`](@ref).

# Keyword Arguments
- `x`: Design matrix for refitting.
- `y`: Response vector for refitting.
- `measure::Union{Symbol,Nothing}=nothing`: Performance measure used to select the
  optimum. Currently only one measure is stored in `SlopeCvResult`.
- `kwargs...`: Additional keyword arguments passed to [`slope`](@ref). User-provided
  keywords override CV-derived defaults.

# Notes
- `x` and `y` are always required to avoid storing mutable training data in
  `SlopeCvResult`.
"""
function refit(
        cv::SlopeCvResult;
        x::Union{AbstractMatrix, SparseMatrixCSC, Nothing} = nothing,
        y::Union{AbstractVector, Nothing} = nothing,
        measure::Union{Symbol, Nothing} = nothing,
        kwargs...,
    )
    user_kwargs = Dict{Symbol, Any}(kwargs)

    has_x = !isnothing(x)
    has_y = !isnothing(y)
    if xor(has_x, has_y) || (!has_x && !has_y)
        throw(ArgumentError("Please provide both `x` and `y` to `refit`."))
    end

    selected_measure = isnothing(measure) ? cv.metric : measure
    if selected_measure != cv.metric
        throw(
            ArgumentError(
                "Measure `$(selected_measure)` not found. Available measures: $(cv.metric)."
            )
        )
    end

    best_γ = cv.best_params["γ"]
    has_explicit_lambda = haskey(user_kwargs, :λ)
    if !isapprox(best_γ, 0.0) && !has_explicit_lambda
        throw(
            ArgumentError(
                "Cannot refit for selected γ=$(best_γ) without explicit `λ`. " *
                "Please pass `λ=...` to `refit`."
            )
        )
    end

    best_q = cv.best_params["q"]
    refit_kwargs = _build_refit_kwargs(best_q, best_α(cv), cv.λ_input, cv.slope_kwargs)
    for (k, v) in user_kwargs
        refit_kwargs[k] = v
    end

    return slope(x::Union{AbstractMatrix, SparseMatrixCSC}, y::AbstractVector; (pairs(refit_kwargs))...)
end

function slopecv_impl(
        x::AbstractMatrix,
        y,
        α,
        λ,
        params::SlopeParameters,
        cv_params::SlopeCvParameters,
    )

    return SLOPE.cv_slope_dense(
        x,
        y,
        α,
        λ,
        params.n,
        params.p,
        params.m,
        params.fit_intercept,
        String(params.loss),
        String(params.λtype),
        String(params.centering),
        String(params.scaling),
        params.path_length,
        params.tol,
        params.max_it,
        params.q,
        params.max_clusters,
        params.dev_change_tol,
        params.dev_ratio_tol,
        params.α_min_ratio,
        String(params.cd_type),
        params.random_seed,
        cv_params.n_folds,
        cv_params.n_repeats,
        String(cv_params.metric),
        StdVector(cv_params.q),
        StdVector(cv_params.γ),
        StdVector(cv_params.fold_indices),
    )
end

function slopecv_impl(
        x::SparseMatrixCSC,
        y,
        α,
        λ,
        params::SlopeParameters,
        cv_params::SlopeCvParameters,
    )

    return SLOPE.cv_slope_sparse(
        x.colptr,
        x.rowval,
        x.nzval,
        y,
        α,
        λ,
        params.n,
        params.p,
        params.m,
        params.fit_intercept,
        String(params.loss),
        String(params.λtype),
        String(params.centering),
        String(params.scaling),
        params.path_length,
        params.tol,
        params.max_it,
        params.q,
        params.max_clusters,
        params.dev_change_tol,
        params.dev_ratio_tol,
        params.α_min_ratio,
        String(params.cd_type),
        params.random_seed,
        cv_params.n_folds,
        cv_params.n_repeats,
        String(cv_params.metric),
        StdVector(cv_params.q),
        StdVector(cv_params.γ),
        StdVector(cv_params.fold_indices),
    )
end

"""
    slopecv(
      x,
      y;
      α=nothing,
      λ=:bh,
      γ=[0.0],
      q=[0.1],
      n_folds=10,
      n_repeats=1,
      metric=:mse,
      kwargs...
    )

Perform cross-validation for SLOPE to find optimal hyperparameters.

# Arguments

- `x::Union{AbstractMatrix,SparseMatrixCSC}`: Input feature matrix, can be dense or sparse.
- `y::AbstractVector`: Response vector.

# Keyword Arguments

- `α::Union{AbstractVector,Real,Nothing}=nothing`: SLOPE regularization path. If `nothing`, it's automatically generated.
- `λ::Union{AbstractVector,Symbol,Nothing}=:bh`: Sequence of regularization
  parameters. If `nothing`, it uses the default BH sequence (`:bh`).
- `γ::Union{AbstractVector,Real}=[0.0]`: Parameter controlling the
  regularization sequence. Multiple values create a grid search.
- `q::Union{AbstractVector}=[0.1]`: FDR parameter for BH sequence. Multiple
  values create a grid search.
- `n_folds::Int=10`: Number of cross-validation folds.
- `n_repeats::Int=1`: Number of times to repeat the CV process with different
  fold assignments.
- `metric::Symbol=:mse`: Evaluation metric for cross-validation. Options
  include `:mse`, `:mae`, `:accuracy`, etc.
- `kwargs...`: Additional parameters passed to the SLOPE solver.

# Returns

A [`SlopeCvResult`](@ref) object.

# Examples
```julia
# Basic usage with default parameters
result = slopecv(X, y)

# Cross-validation with custom parameters
result = slopecv(X, y, γ=[0.0, 0.1, 0.5], q=[0.1, 0.05], n_folds=5, metric=:accuracy)

# Access best parameters and score
best_q = result.best_params["q"]
best_γ = result.best_params["γ"]
best_score = result.best_score
best_α = best_α(result)

# Get final model at the selected CV setting
fit = best_model(result)

# Equivalent explicit refit
fit2 = refit(result, x = X, y = y)
```

# See Also

- [`slope`](@ref): For fitting a SLOPE model with fixed parameters.

"""
function slopecv(
        x::Union{AbstractMatrix, SparseMatrixCSC},
        y::AbstractVector;
        α::Union{AbstractVector, Real, Nothing} = nothing,
        λ::Union{AbstractVector, Symbol, Nothing} = :bh,
        γ::Union{AbstractVector, Real} = [0.0],
        q::Union{AbstractVector} = [0.1],
        n_folds::Int = 10,
        n_repeats::Int = 1,
        metric::Symbol = :mse,
        kwargs...,
    )
    y_input = y
    λ_input = λ
    slope_kwargs = Dict{Symbol, Any}(kwargs)

    params, y, α, λ, original_classes = process_slope_args(
        x,
        y,
        α = α,
        λ = λ,
        kwargs...,
    )

    fold_indices = Int[]

    for _ in 1:n_repeats
        idx = randperm(params.n)
        fold_indices = vcat(fold_indices, idx)
    end

    cv_params = SlopeCvParameters(
        γ,
        q,
        n_folds,
        n_repeats,
        metric,
        fold_indices
    )

    (
        best_score,
        best_ind,
        best_alpha_ind,
        param_name,
        param_value,
        path_lengths,
        scores,
        alphas,
        mean_scores,
        std_errors,
    ) = slopecv_impl(x, y, α, λ, params, cv_params)

    start_idx = 1

    grid_results = Vector{SlopeGridResult}(undef, length(path_lengths))

    param_grid = reshape(param_value, 2, length(path_lengths))

    mean_scores_split = split_by_lengths(mean_scores, path_lengths)
    std_errors_split = split_by_lengths(std_errors, path_lengths)
    alphas_split = split_by_lengths(alphas, path_lengths)

    for i in eachindex(path_lengths)
        n_folds_repeats = cv_params.n_folds * cv_params.n_repeats
        len = path_lengths[i] * n_folds_repeats
        end_idx = start_idx + len - 1

        scores_i = reshape(scores[start_idx:end_idx], n_folds_repeats, :)

        params_i = Dict{String, Any}(
            "q" => param_grid[findfirst(param_name .== 1), i],
            "γ" => param_grid[findfirst(param_name .== 2), i],
        )

        grid_results[i] = SlopeGridResult(
            params_i,
            scores_i,
            alphas_split[i],
            mean_scores_split[i],
            std_errors_split[i],
        )

        start_idx += len
    end

    best_ind_jl = best_ind + 1
    best_alpha_ind_jl = best_alpha_ind + 1

    best_params = grid_results[best_ind_jl].params
    best_α = grid_results[best_ind_jl].alphas[best_alpha_ind_jl]
    best_q = best_params["q"]
    best_γ = best_params["γ"]

    best_fit = _fit_best_model(
        x,
        y_input,
        λ_input,
        best_q,
        best_γ,
        best_α,
        slope_kwargs,
    )

    return SlopeCvResult(
        metric,
        best_score,
        best_ind_jl,
        best_alpha_ind_jl,
        best_params,
        grid_results,
        best_fit,
        λ_input,
        slope_kwargs,
    )

end
