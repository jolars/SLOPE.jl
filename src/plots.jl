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

