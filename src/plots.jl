using SLOPE
using RecipesBase

function coefs(fit::SLOPE.SlopeFit, response=1)
  coef = fit.coefficients
  path_length = length(coef)
  p, m = size(coef[1])
  coef_matrix = zeros(p, path_length)

  if response < 1 || response > m
    error("Response index must be between 1 and $m.")
  end

  for (i, coef) in enumerate(coef)
    for row in 1:p
      coef_matrix[row, i] = coef[row, response]
    end
  end

  return coef_matrix'
end

"""
    plot(fit::SLOPE.SlopeFit; xvar=:α, response=1, kwargs...)

Plot the coefficient paths from a SLOPE model regularization path.

This function visualizes how the coefficients change along the regularization path,
allowing you to see which variables enter the model and how their effects change
as the regularization strength varies.

# Arguments
- `fit::SLOPE.SlopeFit`: A fitted SLOPE model object containing the regularization path
- `xvar::Symbol=:α`: Variable for the x-axis, options:
  - `:α`: Plot against the regularization parameter alpha (default)
  - `:step`: Plot against the step number in the regularization path
- `response::Int=1`: For multi-response models, specifies which response's coefficients to plot
- `layout`: (DEPRECATED) Layout for multi-class plots, e.g., (rows, cols). Default is (m, 1)
  for m classes.

# Keyword Arguments
- `kwargs...`: Additional arguments passed to the plot, such as:
  - `title`: Title for the plot
  - `legend`: Legend position (default: `:none`)
  - `lw`: Line width
  - `color`: Color scheme

# Returns
A plot object showing the coefficient paths

# Examples
```julia
using SLOPE
using Plots

# Fit a SLOPE model
x = randn(100, 20)
y = x[:, 1:5] * ones(5) + randn(100)
fit = slope(x, y)

# Plot coefficient paths against alpha
plot(fit)

# Plot against step number instead
plot(fit, xvar=:step)

# Customize the plot
plot(fit, title="SLOPE Coefficient Paths", lw=2)
```
"""
@recipe function f(fit::SLOPE.SlopeFit; xvar=:α, response=1)
  if xvar == :α
    xscale --> :ln
    xlabel --> "α"
    x = fit.α
  elseif xvar == :step
    xflip --> true
    xlabel --> "Step"
    x = 1:length(fit.α)
  else
    error("Invalid xvar: $xvar. Use :α or :step.")
  end

  legend --> :none
  ylabel --> "β"

  y = coefs(fit, response)

  x, y
end

"""
    plot(cvresult::SlopeCvResult; xvar=:α, index=1; kwargs...)

Plots the cross-validation results from a cross-validated SLOPE model.

# Arguments
- `cvresult::SLOPE.SlopeCvResult`: The result of a cross-validated SLOPE model,
  containing multiple cross-validation results for different parameters.
- `xvar::Symbol=:α`: Variable for the x-axis, options:
  - `:α`: Plot against the regularization parameter alpha (default)
  - `:step`: Plot against the step number in the regularization path
- `index::Int=1`: Index of the cross-validation result to plot. If there are multiple
  values of `γ` or `q`, this specifies which set of these to visualize.

# Keyword Arguments
- `kwargs...`: Additional arguments passed to the plot, such as:
  - `title`: Title for the plot
  - `legend`: Legend position (default: `:none`)
  - `lw`: Line width
  - `color`: Color scheme

# Returns
A plot object showing the cross-validation error, with ribbons for standard error.

# Examples
```julia
using SLOPE, Plots

# Fit a cross-validated SLOPE model
x = randn(100, 20)
y = x[:, 1:5] * ones(5) + randn(100)
cvresult = slopecv(x, y, n_folds=5)

# Plot cross-validation results
plot(cvresult)
```
"""
@recipe function g(cvresult::SLOPE.SlopeCvResult; xvar=:α, index=1)
  if index < 1 || index > length(cvresult.results)
    error("Index must be between 1 and $(length(cvresult.results)).")
  end

  res = cvresult.results[index]

  if xvar == :α
    xscale --> :ln
    xlabel --> "α"
    x = res.alphas
  elseif xvar == :step
    xflip --> true
    xlabel --> "Step"
    x = 1:length(res.alphas)
  else
    error("Invalid xvar: $xvar. Use :α or :step.")
  end

  legend --> :none
  ylabel --> cvresult.metric

  y = res.scores_means

  ribbon --> res.scores_errors

  x, y
end

