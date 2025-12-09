"""
    coef(fit::SlopeFit; index=nothing, simplify=false)

Extract coefficients from a fitted SLOPE model.

# Arguments
- `fit::SlopeFit`: A fitted SLOPE model object
- `index::Union{Int, Nothing}=nothing`: Index of the regularization path point to extract.
  If `nothing` (default), returns coefficients for all path points.
- `simplify::Bool=false`: Whether to simplify the output format:
  - For single response models (`m=1`): Returns a matrix or vector instead of sparse matrices
  - When `index=nothing` and `simplify=true`: Returns a 3D array (p×m×path_length) or
    2D matrix (p×path_length) if `m=1`
  - When `index` is specified and `simplify=true`: Returns a vector if `m=1`, otherwise a matrix

# Returns
- If `index=nothing` and `simplify=false`: Vector of sparse matrices, one per path point
- If `index=nothing` and `simplify=true`: 3D array (p×m×path_length), or 2D if `m=1`
- If `index` is specified and `simplify=false`: Sparse matrix of size p×m
- If `index` is specified and `simplify=true`: Matrix of size p×m, or vector if `m=1`

# Examples
```julia
using SLOPE

# Fit a SLOPE model
x = randn(100, 20)
y = x[:, 1:5] * ones(5) + randn(100)
fit = slope(x, y)

# Get all coefficients along the path (as sparse matrices)
all_coefs = coef(fit)

# Get coefficients at a specific point
coef_10 = coef(fit, index=10)

# Get coefficients as a matrix (for single response)
coef_matrix = coef(fit, simplify=true)  # Returns p×path_length matrix

# Get coefficients at a specific point as a vector (for single response)
coef_vec = coef(fit, index=10, simplify=true)

# Multinomial example
x = randn(150, 10)
y = repeat([1, 2, 3], 50)
fit_multi = slope(x, y, loss=:multinomial)
coef_array = coef(fit_multi, simplify=true)  # Returns p×m×path_length array
```
"""
function coef(fit::SlopeFit; index::Union{Int, Nothing} = nothing, simplify::Bool = false)
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
