"""
    coef(fit::SlopeFit; index=nothing, α=nothing, simplify=false, refit=false)

Extract coefficients from a fitted SLOPE model.

# Arguments
- `fit::SlopeFit`: A fitted SLOPE model object
- `index::Union{Int, Nothing}=nothing`: Index of the regularization path point to extract.
  If `nothing` (default), returns coefficients for all path points.
- `α::Union{Real, Nothing}=nothing`: Specific alpha value to extract coefficients for.
  If the alpha value is not in the fitted path, coefficients are interpolated by default.
  Cannot be used together with `index`.
- `simplify::Bool=false`: Whether to simplify the output format:
  - For single response models (`m=1`): Returns a matrix or vector instead of sparse matrices
  - When `index=nothing` and `simplify=true`: Returns a 3D array (p×m×path_length) or
    2D matrix (p×path_length) if `m=1`
  - When `index` is specified and `simplify=true`: Returns a vector if `m=1`, otherwise a matrix
- `refit::Bool=false`: If `true` and `α` is provided, refit the model at the exact alpha value
  instead of interpolating. Requires that the original data is available (currently not implemented).

# Returns
- If `index=nothing` and `α=nothing` and `simplify=false`: Vector of sparse matrices, one per path point
- If `index=nothing` and `simplify=true`: 3D array (p×m×path_length), or 2D if `m=1`
- If `index` or `α` is specified and `simplify=false`: Sparse matrix of size p×m
- If `index` or `α` is specified and `simplify=true`: Matrix of size p×m, or vector if `m=1`

# Examples
```julia
using SLOPE

# Fit a SLOPE model
x = randn(100, 20)
y = x[:, 1:5] * ones(5) + randn(100)
fit = slope(x, y)

# Get all coefficients along the path (as sparse matrices)
all_coefs = coef(fit)

# Get coefficients at a specific path index
coef_10 = coef(fit, index=10)

# Get coefficients at a specific alpha value (interpolated)
coef_alpha = coef(fit, α=0.5)

# Get coefficients as a matrix (for single response)
coef_matrix = coef(fit, simplify=true)  # Returns p×path_length matrix

# Get coefficients at a specific alpha as a vector (for single response)
coef_vec = coef(fit, α=0.5, simplify=true)

# Multinomial example
x = randn(150, 10)
y = repeat([1, 2, 3], 50)
fit_multi = slope(x, y, loss=:multinomial)
coef_array = coef(fit_multi, simplify=true)  # Returns p×m×path_length array
```
"""
function coef(fit::SlopeFit; index::Union{Int, Nothing} = nothing, 
              α::Union{Real, Nothing} = nothing, 
              simplify::Bool = false, 
              refit::Bool = false)
    
    if !isnothing(index) && !isnothing(α)
        throw(ArgumentError("Cannot specify both `index` and `α`. Choose one."))
    end
    
    if refit && !isnothing(α)
        throw(ArgumentError("`refit=true` is not yet implemented. Only interpolation is supported."))
    end
    
    # Handle alpha parameter
    if !isnothing(α)
        return _coef_at_alpha(fit, α, simplify)
    end
    
    coefs = fit.coefficients
    path_length = length(coefs)

    p, m = size(coefs[1])

    return if isnothing(index)
        if simplify
            coef_array = zeros(p, m, path_length)
            for (i, coef) in enumerate(coefs)
                coef_array[:, :, i] = coef
            end

            if m == 1
                return dropdims(coef_array, dims = 2)
            else
                return coef_array
            end
        else
            return coefs
        end
    elseif isa(index, Int)
        if index < 1 || index > path_length
            throw(BoundsError("Index $index out of bounds. Must be between 1 and $path_length."))
        end

        coefs_at_index = coefs[index]

        if simplify && m == 1
            return vec(coefs_at_index)
        else
            return coefs_at_index
        end
    else
        throw(ArgumentError("Index must be an integer or nothing."))
    end
end

"""
    _coef_at_alpha(fit::SlopeFit, alpha::Real, simplify::Bool)

Internal function to extract or interpolate coefficients at a specific alpha value.
"""
function _coef_at_alpha(fit::SlopeFit, alpha::Real, simplify::Bool)
    alphas = fit.α
    
    if alpha < minimum(alphas) || alpha > maximum(alphas)
        throw(ArgumentError("alpha=$alpha is outside the fitted path range [$(minimum(alphas)), $(maximum(alphas))]"))
    end
    
    # Check if alpha is exactly in the path
    idx = findfirst(≈(alpha), alphas)
    if !isnothing(idx)
        return coef(fit, index=idx, simplify=simplify)
    end
    
    # Find bracketing alphas for interpolation
    # Alpha decreases along the path, so we need the indices where alpha is between two values
    idx_upper = findfirst(α -> α <= alpha, alphas)
    
    if isnothing(idx_upper) || idx_upper == 1
        # Alpha is at or beyond the first point
        return coef(fit, index=1, simplify=simplify)
    end
    
    idx_lower = idx_upper - 1
    alpha_lower = alphas[idx_lower]
    alpha_upper = alphas[idx_upper]
    
    # Linear interpolation weight
    weight = (alpha - alpha_upper) / (alpha_lower - alpha_upper)
    
    # Get coefficients at both points
    coef_lower = Matrix(fit.coefficients[idx_lower])
    coef_upper = Matrix(fit.coefficients[idx_upper])
    
    # Interpolate
    coef_interp = weight * coef_lower + (1 - weight) * coef_upper
    
    # Return in appropriate format
    p, m = size(coef_interp)
    if simplify && m == 1
        return vec(coef_interp)
    else
        return coef_interp
    end
end
