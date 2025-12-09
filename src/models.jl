using SparseArrays

struct SlopeParameters
  n::Int
  p::Int
  m::Int
  fit_intercept::Bool
  loss::Symbol
  centering::Symbol
  scaling::Symbol
  path_length::Int
  tol::Real
  max_it::Int
  q::Union{AbstractVector,Real}
  max_clusters::Union{Int,Nothing}
  dev_change_tol::Real
  dev_ratio_tol::Real
  α_min_ratio::Real
  cd_type::Symbol
  random_seed::Int
end

function process_slope_args(
  x::Union{AbstractMatrix,SparseMatrixCSC},
  y::AbstractVector;
  α::Union{AbstractVector,Real,Nothing}=nothing,
  λ::Union{AbstractVector,Nothing,Symbol}=:bh,
  fit_intercept::Bool=true,
  loss::Symbol=:quadratic,
  centering::Symbol=:mean,
  scaling::Symbol=:sd,
  path_length::Int=100,
  tol::Float64=1e-5,
  max_it::Int=10000,
  q::Float64=0.1,
  max_clusters::Union{Int,Nothing}=nothing,
  dev_change_tol::Float64=1e-5,
  dev_ratio_tol::Float64=0.999,
  α_min_ratio::Union{Float64,Nothing}=nothing,
  cd_type::Symbol=:permuted,
  random_seed::Int=0,
)
  n, p = size(x)
  m = 1

  y = convert(Array{Float64}, y)

  unique_classes = nothing

  if loss == :multinomial
    unique_classes = sort(unique(y))
    n_classes = length(unique_classes)
    m = n_classes - 1

    # Create a mapping from original classes to 0-based consecutive integers
    class_map = Dict(class => i - 1 for (i, class) in enumerate(unique_classes))

    # Transform input classes using the mapping
    y = [class_map[class] for class in y]
    y = convert(Array{Float64}, y)
  end

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

  if isa(λ, Symbol)
    λ = regweights(p*m; q=q, type=λ, n=n)
  end

  if isnothing(α_min_ratio)
    α_min_ratio = n > p * m ? 1e-2 : 1e-4
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
    cd_type,
    random_seed,
  )

  return params, y, α, λ, unique_classes
end

"""
    SlopeFit

A structure containing the results of fitting a SLOPE model.

# Fields

- `intercepts::Vector{Vector{Float64}}`: A vector of intercept vectors along the regularization path.
  For each point in the path, contains a vector of length `m` with class-specific intercepts.
- `coefficients::Vector{SparseMatrixCSC{Float64,Int}}`: A vector of sparse coefficient matrices 
  along the regularization path. Each matrix is of size `p×m` where `p` is the number of 
  predictors and `m` is the number of response classes (1 for regression).
- `α::Vector{Float64}`: The alpha values used at each point of the regularization path.
- `λ::Vector{Float64}`: The lambda values used at each point of the regularization path.
- `m::Int`: The number of response classes (1 for regression, >1 for multinomial).
- `loss::Symbol`: The loss function used in the model fitting process.
- `classes::Union{Vector,Nothing}`: A vector of unique class labels for the
  response variable. This is `nothing` for regression models (continuous responses).
"""
struct SlopeFit
  intercepts::Vector{Vector{Float64}}
  coefficients::Vector{SparseMatrixCSC{Float64,Int}}
  α::Vector{Float64}
  λ::Vector{Float64}
  m::Int
  loss::Symbol
  classes::Union{Vector,Nothing}
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
    String(params.loss),
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

  SLOPE.fit_slope_sparse(
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
- `λ::Union{AbstractVector,Symbol,Nothing}=:bh`: Lambda sequence for regularization path. Can
  be a vector or a symbol indicating the type of sequence to generate. If `nothing`, the value `:bh` is used.
- `fit_intercept::Bool=true`: Whether to fit an intercept term
- `loss::Symbol=:quadratic`: Type of loss function
- `centering::Symbol=:mean`: Method for centering predictors
- `scaling::Symbol=:sd`: Method for scaling predictors
- `path_length::Int=100`: Number of regularization path points
- `tol::Float64=1e-5`: Convergence tolerance for optimization
- `max_it::Int=10000`: Maximum number of iterations
- `q::Float64=0.1`: Parameter controlling the shape of the penalty
  weights sequence. Should be in the range (0, 1). 
- `max_clusters::Union{Int,Nothing}=nothing`: Early path stopping
  criteria for maximum number of clusters (defaults to n+1) 
- `dev_change_tol::Float64=1e-5`: Early path stopping criteria for tolerance for
  change in deviance 
- `dev_ratio_tol::Float64=0.999`: Early path stopping
  criteria for tolerance for ratio of deviance
- `α_min_ratio::Union{Float64,Nothing}=nothing`: Fraction of maximum `α` to use as minimum
  value in the regularization path. Defaults to `1e-2` if `n > p * m`, otherwise `1e-4`.

# Returns
A [`SlopeFit`](@ref) object.

"""
function slope(
  x::Union{AbstractMatrix,SparseMatrixCSC},
  y::AbstractVector;
  α::Union{AbstractVector,Real,Nothing}=nothing,
  λ::Union{AbstractVector,Nothing,Symbol}=:bh,
  fit_intercept::Bool=true,
  loss::Symbol=:quadratic,
  centering::Symbol=:mean,
  scaling::Symbol=:sd,
  path_length::Int=100,
  tol::Float64=1e-5,
  max_it::Int=10000,
  q::Float64=0.1,
  max_clusters::Union{Int,Nothing}=nothing,
  dev_change_tol::Float64=1e-5,
  dev_ratio_tol::Float64=0.999,
  α_min_ratio::Union{Float64,Nothing}=nothing,
  cd_type::Symbol=:permuted,
  random_seed::Int=0,
)

  params, y, α, λ, original_classes = process_slope_args(
    x,
    y,
    α=α,
    λ=λ,
    fit_intercept=fit_intercept,
    loss=loss,
    centering=centering,
    scaling=scaling,
    path_length=path_length,
    tol=tol,
    max_it=max_it,
    q=q,
    max_clusters=max_clusters,
    dev_change_tol=dev_change_tol,
    dev_ratio_tol=dev_ratio_tol,
    α_min_ratio=α_min_ratio,
    cd_type=cd_type,
    random_seed=random_seed,
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
  intercept_vectors = Vector{Vector{Float64}}()

  if !isempty(intercepts)
    # Reshape into a matrix and convert each row to a vector
    intercept_matrix = reshape(intercepts, params.m, :)'  # path_length × m matrix
    for i in axes(intercept_matrix, 1)
      push!(intercept_vectors, intercept_matrix[i, :])
    end
  end

  for i in eachindex(nnz)
    if ind > nnz[i]
      empty_mat = spzeros(Float64, params.p, params.m)
      push!(coefs, empty_mat)
      continue
    end

    rng = ind:nnz[i]
    coefs_step = sparse(coef_rows[rng], coef_cols[rng], coef_vals[rng], params.p, params.m)
    push!(coefs, coefs_step)
    ind = nnz[i] + 1
  end

  SlopeFit(
    intercept_vectors,
    coefs,
    alpha_out,
    lambda_out,
    params.m,
    loss,
    original_classes
  )
end

function predict(fit::SlopeFit, x::Union{AbstractMatrix,SparseMatrixCSC})
  path_length = length(fit.α)
  predictions = Vector{Matrix{Float64}}(undef, path_length)

  for i in 1:path_length
    eta = Matrix(x * fit.coefficients[i] .+ fit.intercepts[i])
    n, m = size(eta)
    pred, pred_cols = SLOPE.slope_predict(eta, n, m, String(fit.loss))
    predictions[i] = reshape(pred, n, Int64(pred_cols))
  end

  return predictions
end
