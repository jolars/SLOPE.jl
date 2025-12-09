using Distributions
using StatsBase

"""
    regweights(
      p::Int;
      q::Float64=0.1,
      type::Symbol=:bh,
      n=nothing,
      θ1::Real=1.0,
      θ2::Real=1.0
    )

Generates a sequence of regularization weights for the sorted L1 norm.

# Arguments
- `p::Int`: The number of lambda values to generate (number of features)

# Keyword Arguments
- `q::Float64=0.1`: The false discovery rate (FDR) level or quantile value (must be in (0, 1))
- `type::Symbol=:bh`: The type of sequence to generate:
  - `:bh`: Benjamini-Hochberg sequence
  - `:gaussian`: Gaussian sequence
  - `:oscar`: Octagonal Shrinkage and Clustering Algorithm for Regression
  - `:lasso`: Lasso (all weights equal to 1.0)
- `n::Union{Int, Nothing}=nothing`: Number of observations (required for `:gaussian` type)
- `θ1::Real=1.0`: First parameter for OSCAR weights
- `θ2::Real=1.0`: Second parameter for OSCAR weights

# Returns
- `Vector{Float64}`: The generated lambda sequence in decreasing order

# Examples
```julia
λ = regweights(100)  # Benjamini-Hochberg with default q=0.1
λ = regweights(100, q=0.05, type=:gaussian, n=200)
λ = regweights(50, type=:oscar, θ1=1.5, θ2=0.5)
```
"""
function regweights(
        p::Int;
        q::Float64 = 0.1,
        type::Symbol = :bh,
        n::Union{Int, Nothing} = nothing,
        θ1::Real = 1.0,
        θ2::Real = 1.0,
    )
    if p <= 0
        throw(DomainError(p, "p must be positive"))
    end

    λ = zeros(Float64, p)

    valid_types = [:bh, :gaussian, :oscar, :lasso]
    if !(type in valid_types)
        throw(ArgumentError("Invalid type: $type. Must be one of $(valid_types)."))
    end

    if type == :gaussian && isnothing(n)
        throw(ArgumentError("n must be provided for type 'gaussian'"))
    end

    if type in [:bh, :gaussian]
        if q <= 0 || q >= 1
            throw(DomainError(q, "q must be between 0 and 1"))
        end

        for j in 1:p
            λ[j] = quantile(Normal(), 1.0 - (j) * q / (2.0 * p))
        end

        if type == :gaussian && p > 1
            if n <= 0
                throw(DomainError(n, "n must be positive for type :gaussian"))
            end

            sum_sq = 0.0
            for i in 2:p
                sum_sq += λ[i - 1]^2
                w = 1.0 / max(1.0, n - i - 1.0)
                λ[i] *= sqrt(1.0 + w * sum_sq)
            end

            # Ensure non-increasing λ
            for i in 2:p
                if λ[i] > λ[i - 1]
                    λ[i] = λ[i - 1]
                end
            end
        end
    elseif type == :oscar
        if θ1 < 0
            throw(DomainError(θ1, "θ1 must be non-negative"))
        end

        if θ2 < 0
            throw(DomainError(θ2, "θ2 must be non-negative"))
        end

        λ .= θ1 .+ θ2 .* (p .- collect(1:p))
    elseif type == :lasso
        λ .= 1.0
    end

    @assert minimum(λ) > 0 "lambda must be positive"
    @assert all(isfinite, λ) "lambda must be finite"
    @assert length(λ) == p "lambda sequence is of right size"

    return λ
end
