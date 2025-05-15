using SparseArrays

struct SlopeParameters
  n::Int
  p::Int
  m::Int
  fit_intercept::Bool
  loss::String
  centering::String
  scaling::String
  path_length::Int
  tol::Real
  max_it::Int
  q::Real
  max_clusters::Union{Int,Nothing}
  dev_change_tol::Real
  dev_ratio_tol::Real
  α_min_ratio::Real
end

function fitslope(
  x::AbstractMatrix,
  y,
  α,
  λ,
  params::SlopeParameters,
  coef_vals,
  coef_rows,
  coef_cols,
  intercepts,
  nnz,
  alpha_out,
  lambda_out,
)
  SLOPE.fit_slope_dense(
    x,
    y,
    α,
    λ,
    params.n,
    params.p,
    params.m,
    params.fit_intercept,
    params.loss,
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
    coef_vals,
    coef_rows,
    coef_cols,
    intercepts,
    nnz,
    alpha_out,
    lambda_out
  )
end

function fitslope(
  x::SparseMatrixCSC,
  y,
  α,
  λ,
  params::SlopeParameters,
  coef_vals,
  coef_rows,
  coef_cols,
  intercepts,
  nnz,
  alpha_out,
  lambda_out,
)
  x_rows, x_cols, x_vals = findnz(x)

  SLOPE.fit_slope_sparse(
    x_rows,
    x_cols,
    x_vals,
    y,
    α,
    λ,
    params.n,
    params.p,
    params.m,
    params.fit_intercept,
    params.loss,
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
    coef_vals,
    coef_rows,
    coef_cols,
    intercepts,
    nnz,
    alpha_out,
    lambda_out
  )
end

"""
    slope(x, y; kwargs...) -> NamedTuple

Fit a SLOPE (Sorted L1 Penalized Estimation) model to the provided data.

SLOPE is a regularization method that combines the L1 norm with a sorted penalty,
encouraging both sparsity and grouping of features.

# Arguments
- `x`: Matrix of predictors (dense or sparse)
- `y`: Response variable (vector)

# Keyword Arguments
- `α::Union{AbstractVector,Real,Nothing}=nothing`: Alpha sequence for regularization path
- `λ::Union{AbstractVector,Nothing}=nothing`: Lambda sequence for regularization path
- `fit_intercept::Bool=true`: Whether to fit an intercept term
- `loss::String="quadratic"`: Type of loss function
- `centering::String="mean"`: Method for centering predictors
- `scaling::String="sd"`: Method for scaling predictors
- `path_length::Int=100`: Number of regularization path points
- `tol::Float64=1e-5`: Convergence tolerance for optimization
- `max_it::Int=10000`: Maximum number of iterations
- `q::Float64=0.1`: Parameter for regularization sequence
- `max_clusters::Union{Int,Nothing}=nothing`: Maximum number of clusters (defaults to n+1)
- `dev_change_tol::Float64=1e-5`: Tolerance for change in deviance
- `dev_ratio_tol::Float64=0.999`: Tolerance for ratio of deviance
- `α_min_ratio::Union{Float64,Nothing}=nothing`: Minimum alpha ratio for regularization path

# Returns
A `NamedTuple` containing:
- `β`: Array of sparse coefficient matrices along the regularization path
- `β0`: Array of intercept values along the regularization path
- `α`: Effective alpha values used
- `λ`: Effective lambda values used
"""
function slope(
  x::Union{AbstractMatrix,SparseMatrixCSC},
  y::AbstractVector;
  α::Union{AbstractVector,Real,Nothing}=nothing,
  λ::Union{AbstractVector,Nothing}=nothing,
  fit_intercept::Bool=true,
  loss::String="quadratic",
  centering::String="mean",
  scaling::String="sd",
  path_length::Int=100,
  tol::Float64=1e-5,
  max_it::Int=10000,
  q::Float64=0.1,
  max_clusters::Union{Int,Nothing}=nothing,
  dev_change_tol::Float64=1e-5,
  dev_ratio_tol::Float64=0.999,
  α_min_ratio::Union{Float64,Nothing}=nothing,
)
  n, p = size(x)

  m = 1 # FIXME: This should be the number of groups

  if isnothing(max_clusters)
    max_clusters = n + 1
  end

  if isnothing(α)
    α = Float64[]
  elseif isa(α, Real)
    α = [α]
  end

  if isnothing(λ)
    λ = Float64[]
  end

  if isnothing(α_min_ratio)
    α_min_ratio = n > p ? 1e-2 : 1e-4
  end

  params = SlopeParameters(
    n,
    p,
    m,
    fit_intercept,
    loss,
    centering,
    scaling,
    path_length,
    tol,
    max_it,
    q,
    max_clusters,
    dev_change_tol,
    dev_ratio_tol,
    α_min_ratio,
  )

  coef_vals = Float64[]
  coef_rows = Int[]
  coef_cols = Int[]
  nnz = Int[]

  intercepts = Float64[]

  alpha_out = Float64[]
  lambda_out = Float64[]

  fitslope(
    x,
    y,
    α,
    λ,
    params,
    coef_vals,
    coef_rows,
    coef_cols,
    intercepts,
    nnz,
    alpha_out,
    lambda_out,
  )

  ind = 1

  coefs = []

  for i in eachindex(nnz)
    if ind > nnz[i]
      empty_mat = spzeros(Float64, p, m)
      push!(coefs, empty_mat)
      continue
    end

    rng = ind:nnz[i]
    coefs_step = sparse(coef_rows[rng], coef_cols[rng], coef_vals[rng])
    push!(coefs, coefs_step)
    ind = nnz[i] + 1
  end

  (β=coefs, β0=intercepts, α=alpha_out, λ=lambda_out)
end
