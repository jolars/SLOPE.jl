using SparseArrays

struct SlopeParameters
    n::Int
    p::Int
    m::Int
    fit_intercept::Bool
    loss::Symbol
    λtype::Symbol
    centering::Symbol
    scaling::Symbol
    path_length::Int
    tol::Real
    max_it::Int
    q::Union{AbstractVector, Real}
    max_clusters::Union{Int, Nothing}
    dev_change_tol::Real
    dev_ratio_tol::Real
    α_min_ratio::Real
    cd_type::Symbol
    random_seed::Int
end

function process_slope_args(
        x::Union{AbstractMatrix, SparseMatrixCSC},
        y::AbstractVector;
        α::Union{AbstractVector, Real, Nothing} = nothing,
        λ::Union{AbstractVector, Nothing, Symbol} = :bh,
        fit_intercept::Bool = true,
        loss::Symbol = :quadratic,
        centering::Symbol = :mean,
        scaling::Symbol = :sd,
        path_length::Int = 100,
        tol::Float64 = 1.0e-5,
        max_it::Int = 10000,
        q::Float64 = 0.1,
        max_clusters::Union{Int, Nothing} = nothing,
        dev_change_tol::Float64 = 1.0e-5,
        dev_ratio_tol::Float64 = 0.999,
        α_min_ratio::Union{Float64, Nothing} = nothing,
        cd_type::Symbol = :permuted,
        random_seed::Int = 0,
    )
    n, p = size(x)
    m = 1

    y = convert(Array{Float64}, y)

    unique_classes = nothing

    λtype = :bh

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
        λtype = λ
        λ = regweights(p * m; q = q, type = λ, n = n)
    end

    if isnothing(α_min_ratio)
        α_min_ratio = n > p * m ? 1.0e-2 : 1.0e-4
    end

    params = SlopeParameters(
        n,
        p,
        m,
        fit_intercept,
        loss,
        λtype,
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

# Examples
```julia
using SLOPE

# Fit a SLOPE model
x = randn(100, 10)
y = randn(100)
fit = slope(x, y)

# Access results
fit.coefficients[1]  # Coefficients at first point of regularization path
fit.α  # Alpha values along the path
fit.intercepts[end]  # Intercept at last point of regularization path
```
"""
struct SlopeFit
    intercepts::Vector{Vector{Float64}}
    coefficients::Vector{SparseMatrixCSC{Float64, Int}}
    α::Vector{Float64}
    λ::Vector{Float64}
    m::Int
    loss::Symbol
    classes::Union{Vector, Nothing}
end

function Base.show(io::IO, ::MIME"text/plain", fit::SlopeFit)
    n_solutions = length(fit.α)
    p = size(fit.coefficients[1], 1)
    has_intercept = !isempty(fit.intercepts)

    # Count non-zero coefficients for each solution
    n_nonzero = [nnz(coef) for coef in fit.coefficients]

    # Family name mapping
    family_map = Dict(
        :quadratic => "gaussian",
        :logistic => "binomial",
        :multinomial => "multinomial",
        :poisson => "poisson"
    )
    family = get(family_map, fit.loss, string(fit.loss))

    println(io, "SLOPE fit")
    println(io)
    println(io, "Family: ", family)
    println(io, "Predictors: ", p)
    println(io, "Intercept: ", has_intercept ? "Yes" : "No")
    if fit.m > 1
        println(io, "Classes: ", fit.m)
    end
    println(io)
    println(io, "Regularization path:")
    println(io, "  Length: ", n_solutions, " steps")
    println(io, "  Alpha range: ", round(minimum(fit.α), sigdigits = 3), " to ", round(maximum(fit.α), sigdigits = 3))
    println(io)

    # Show first and last 5 steps
    n_show = min(5, n_solutions)
    println(io, "Path summary (first and last ", n_show, " steps):")

    # Header
    println(io, "  ", rpad("alpha", 12), rpad("n_nonzero", 12))

    # First n_show
    for i in 1:n_show
        println(io, "  ", rpad(round(fit.α[i], sigdigits = 3), 12), rpad(n_nonzero[i], 12))
    end

    # Ellipsis if there are more than 2*n_show solutions
    if n_solutions > 2 * n_show
        println(io, "  ...")
    end

    # Last n_show
    return if n_solutions > n_show
        start_idx = max(n_show + 1, n_solutions - n_show + 1)
        for i in start_idx:(n_solutions - 1)
            println(io, "  ", rpad(round(fit.α[i], sigdigits = 3), 12), rpad(n_nonzero[i], 12))
        end
        # Last line without newline
        print(io, "  ", rpad(round(fit.α[n_solutions], sigdigits = 3), 12), rpad(n_nonzero[n_solutions], 12))
    end
end

# Compact one-line display for inline printing
function Base.show(io::IO, fit::SlopeFit)
    n_solutions = length(fit.α)
    p = size(fit.coefficients[1], 1)

    print(io, "SlopeFit(")
    print(io, "loss=", fit.loss, ", ")
    print(io, "n_solutions=", n_solutions, ", ")
    print(io, "n_features=", p, ", ")
    print(io, "n_classes=", fit.m)
    return print(io, ")")
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
    return SLOPE.fit_slope_dense(
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

    return SLOPE.fit_slope_sparse(
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
    slope(x, y; kwargs...) -> SlopeFit

Fit a SLOPE (Sorted L1 Penalized Estimation) model to the provided data.

SLOPE is a regularization method that combines the L1 norm with a sorted penalty,
encouraging both sparsity and grouping of features.

# Arguments
- `x`: Matrix of predictors (dense or sparse)
- `y`: Response variable (vector)

# Keyword Arguments
- `α::Union{AbstractVector,Real,Nothing}=nothing`: Alpha sequence for regularization path.
  If `nothing`, a sequence is automatically generated.
- `λ::Union{AbstractVector,Symbol,Nothing}=:bh`: Lambda (penalty weights) sequence. Can
  be a vector of custom weights, or a symbol specifying the sequence type:
  - `:bh`: Benjamini-Hochberg sequence
  - `:gaussian`: Gaussian sequence
  - `:oscar`: Octagonal Shrinkage and Clustering Algorithm for Regression
  - `:lasso`: Lasso (all weights equal to 1.0)
  If `nothing`, defaults to `:bh`.
- `fit_intercept::Bool=true`: Whether to fit an intercept term
- `loss::Symbol=:quadratic`: Loss function type. Options: 
  - `:quadratic`: Gaussian (continuous response)
  - `:logistic`: Binomial (binary classification)
  - `:multinomial`: Multinomial (multi-class classification)
  - `:poisson`: Poisson (count data)
- `centering::Symbol=:mean`: Predictor centering method. Options: `:mean` (center by mean), 
  `:none` (no centering)
- `scaling::Symbol=:sd`: Predictor scaling method. Options: `:sd` (scale by standard deviation), 
  `:none` (no scaling)
- `path_length::Int=100`: Number of regularization path points
- `tol::Float64=1e-5`: Convergence tolerance for optimization
- `max_it::Int=10000`: Maximum number of iterations
- `q::Float64=0.1`: Parameter controlling the shape of the penalty weights sequence. 
  Should be in the range (0, 1).
- `max_clusters::Union{Int,Nothing}=nothing`: Early path stopping criterion for maximum 
  number of clusters (defaults to n+1)
- `dev_change_tol::Float64=1e-5`: Early path stopping criterion for tolerance of change in deviance
- `dev_ratio_tol::Float64=0.999`: Early path stopping criterion for tolerance of deviance ratio
- `α_min_ratio::Union{Float64,Nothing}=nothing`: Fraction of maximum `α` to use as minimum
  value in the regularization path. Defaults to `1e-2` if `n > p * m`, otherwise `1e-4`.
- `cd_type::Symbol=:permuted`: Coordinate descent update type. Options: `:permuted` (random 
  permutation order), `:cyclic` (sequential order)
- `random_seed::Int=0`: Random seed for reproducibility

# Returns
A [`SlopeFit`](@ref) object.

# Examples
```julia
using SLOPE

# Basic regression example
x = randn(100, 20)
y = x[:, 1:5] * ones(5) + randn(100)
fit = slope(x, y)

# Use custom regularization parameters
fit = slope(x, y, q=0.05, path_length=50)

# Use OSCAR-type lambda sequence
fit = slope(x, y, λ=:oscar)

# Logistic regression
x = randn(100, 10)
y = rand([0, 1], 100)
fit = slope(x, y, loss=:logistic)

# Poisson regression (count data)
x = randn(100, 10)
y = rand(0:10, 100)
fit = slope(x, y, loss=:poisson)

# Multinomial classification
x = randn(150, 10)
y = repeat([1, 2, 3], 50)
fit = slope(x, y, loss=:multinomial)
```
"""
function slope(
        x::Union{AbstractMatrix, SparseMatrixCSC},
        y::AbstractVector;
        α::Union{AbstractVector, Real, Nothing} = nothing,
        λ::Union{AbstractVector, Nothing, Symbol} = :bh,
        fit_intercept::Bool = true,
        loss::Symbol = :quadratic,
        centering::Symbol = :mean,
        scaling::Symbol = :sd,
        path_length::Int = 100,
        tol::Float64 = 1.0e-5,
        max_it::Int = 10000,
        q::Float64 = 0.1,
        max_clusters::Union{Int, Nothing} = nothing,
        dev_change_tol::Float64 = 1.0e-5,
        dev_ratio_tol::Float64 = 0.999,
        α_min_ratio::Union{Float64, Nothing} = nothing,
        cd_type::Symbol = :permuted,
        random_seed::Int = 0,
    )

    params, y, α, λ, original_classes = process_slope_args(
        x,
        y,
        α = α,
        λ = λ,
        fit_intercept = fit_intercept,
        loss = loss,
        centering = centering,
        scaling = scaling,
        path_length = path_length,
        tol = tol,
        max_it = max_it,
        q = q,
        max_clusters = max_clusters,
        dev_change_tol = dev_change_tol,
        dev_ratio_tol = dev_ratio_tol,
        α_min_ratio = α_min_ratio,
        cd_type = cd_type,
        random_seed = random_seed,
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

    return SlopeFit(
        intercept_vectors,
        coefs,
        alpha_out,
        lambda_out,
        params.m,
        loss,
        original_classes
    )
end

"""
    predict(fit::SlopeFit, x) -> Vector{Matrix{Float64}}

Generate predictions from a fitted SLOPE model.

Returns predictions for each point along the regularization path. For regression models,
predictions are continuous values. For classification models (logistic, multinomial), 
predictions are class probabilities or predicted class labels depending on the loss function.

# Arguments
- `fit::SlopeFit`: A fitted SLOPE model from [`slope`](@ref)
- `x::Union{AbstractMatrix,SparseMatrixCSC}`: Predictor matrix (n × p)

# Returns
A vector of prediction matrices, one for each point in the regularization path.
Each matrix has dimensions (n × m), where n is the number of observations and m 
depends on the model type:
- For regression (`:quadratic`): m = 1 (predicted values)
- For Poisson (`:poisson`): m = 1 (predicted counts/rates)
- For logistic (`:logistic`): m = 1 (predicted probabilities)
- For multinomial (`:multinomial`): m = number of classes (class probabilities)

# Examples
```julia
using SLOPE

# Regression
x = randn(100, 10)
y = randn(100)
fit = slope(x, y)
x_new = randn(20, 10)
predictions = predict(fit, x_new)
predictions[1]  # Predictions at first regularization point

# Classification
x = randn(100, 10)
y = rand([0, 1], 100)
fit = slope(x, y, loss=:logistic)
predictions = predict(fit, x_new)
predictions[end]  # Predictions at last regularization point
```
"""
function predict(fit::SlopeFit, x::Union{AbstractMatrix, SparseMatrixCSC})
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
