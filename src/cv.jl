using CxxWrap
using Random

struct SlopeCvParameters
  γ::Union{Float64,AbstractVector}
  q::Union{Float64,AbstractVector}
  n_folds::Int
  n_repeats::Int
  metric::String
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
  params::Dict{String,Any}
  scores::Matrix{Real}
  alphas::Vector{Real}
  scores_means::Vector{Real}
  scores_errors::Vector{Real}
end

"""
    SlopeCvResult

Result structure from SLOPE cross-validation.

# Fields
- `metric::String`: The evaluation metric used (e.g., "mse", "accuracy")
- `best_score::Real`: The best score achieved during cross-validation
- `best_ind::Int`: Index of the best parameter combination
- `best_α_ind::Int`: Index of the best alpha value in the regularization path
- `best_params::Dict{String,Any}`: Dictionary with the best parameter values
- `results::Vector{SlopeGridResult}`: Grid search results for each parameter combination
"""
struct SlopeCvResult
  metric::String
  best_score::Real
  best_ind::Int
  best_α_ind::Int
  best_params::Dict{String,Any}
  results::Vector{SlopeGridResult}
end

function slopecv_impl(
  x::AbstractMatrix,
  y,
  α,
  λ,
  params::SlopeParameters,
  cv_params::SlopeCvParameters,
)

  SLOPE.cv_slope_dense(
    x,
    y,
    α,
    λ,
    params.n,
    params.p,
    params.m,
    params.fit_intercept,
    String(params.loss),
    params.centering,
    params.scaling,
    params.path_length,
    params.tol,
    params.max_it,
    params.q,
    params.max_clusters,
    params.dev_change_tol,
    params.dev_ratio_tol,
    params.α_min_ratio,
    cv_params.n_folds,
    cv_params.n_repeats,
    cv_params.metric,
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

  SLOPE.cv_slope_sparse(
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
    params.centering,
    params.scaling,
    params.path_length,
    params.tol,
    params.max_it,
    params.q,
    params.max_clusters,
    params.dev_change_tol,
    params.dev_ratio_tol,
    params.α_min_ratio,
    cv_params.n_folds,
    cv_params.n_repeats,
    cv_params.metric,
    StdVector(cv_params.q),
    StdVector(cv_params.γ),
    StdVector(cv_params.fold_indices),
  )
end

"""
    slopecv(x, y; α=nothing, λ=nothing, γ=[0.0], q=[0.1], n_folds=10, n_repeats=1, metric="mse", kwargs...)

Perform cross-validation for the SLOPE method to find optimal hyperparameters.

# Arguments
- `x::Union{AbstractMatrix,SparseMatrixCSC}`: Input feature matrix, can be dense or sparse.
- `y::AbstractVector`: Target vector.

# Keyword Arguments
- `α::Union{AbstractVector,Real,Nothing}=nothing`: SLOPE regularization path. If `nothing`, it's automatically generated.
- `λ::Union{AbstractVector,Nothing}=nothing`: Sequence of regularization parameters. If `nothing`, it's automatically generated.
- `γ::Union{AbstractVector,Real}=[0.0]`: Parameter controlling the regularization sequence. Multiple values create a grid search.
- `q::Union{AbstractVector}=[0.1]`: FDR parameter for BH sequence. Multiple values create a grid search.
- `n_folds::Int=10`: Number of cross-validation folds.
- `n_repeats::Int=1`: Number of times to repeat the CV process with different fold assignments.
- `metric::String="mse"`: Evaluation metric for cross-validation. Options include "mse", "mae", "accuracy", etc.
- `kwargs...`: Additional parameters passed to the SLOPE solver.

# Returns
`SlopeCvResult`: A structure containing:
- `metric`: The evaluation metric used
- `best_score`: The best score achieved during CV
- `best_ind`: Index of the best parameter combination
- `best_α_ind`: Index of the best alpha value
- `best_params`: Dictionary with the best parameter values
- `results`: Vector of `SlopeGridResult` for each parameter combination

# Examples
```julia
# Basic usage with default parameters
result = slopecv(X, y)

# Cross-validation with custom parameters
result = slopecv(X, y, γ=[0.0, 0.1, 0.5], q=[0.1, 0.05], n_folds=5, metric="accuracy")

# Access best parameters and score
best_q = result.best_params["q"]
best_γ = result.best_params["γ"]
best_score = result.best_score
```

# See Also
- `fit_slope`: For fitting a SLOPE model with fixed parameters.
- `SlopeParameters`: Structure defining parameters for the SLOPE algorithm.
"""
function slopecv(
  x::Union{AbstractMatrix,SparseMatrixCSC},
  y::AbstractVector;
  α::Union{AbstractVector,Real,Nothing}=nothing,
  λ::Union{AbstractVector,Nothing}=nothing,
  γ::Union{AbstractVector,Real}=[0.0],
  q::Union{AbstractVector}=[0.1],
  n_folds::Int=10,
  n_repeats::Int=1,
  metric::String="mse",
  kwargs...,
)
  params, y, α, λ, original_classes = process_slope_args(
    x,
    y,
    α=α,
    λ=λ,
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

    params_i = Dict{String,Any}(
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

  best_params = grid_results[best_ind+1].params

  SlopeCvResult(
    metric,
    best_score,
    best_ind,
    best_alpha_ind,
    best_params,
    grid_results,
  )

end

