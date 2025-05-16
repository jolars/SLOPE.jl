module PlotSLOPE

using SLOPE
using Plots

"""
    plot(fit::SLOPE.SlopeFit; xvar::Symbol=:α, layout=nothing, kwargs...)

Plot the coefficient paths from a SLOPE model regularization path.

This function visualizes how the coefficients change along the regularization path,
allowing you to see which variables enter the model and how their effects change
as the regularization strength varies.

For multinomial models (when fit.m > 1), this creates a multi-panel plot with
one panel per response class.

# Arguments
- `fit::SLOPE.SlopeFit`: A fitted SLOPE model object containing the regularization path
- `xvar::Symbol=:α`: Variable for the x-axis, options:
  - `:α`: Plot against the regularization parameter alpha (default)
  - `:step`: Plot against the step number in the regularization path
- `layout`: Layout for multi-class plots, e.g., (rows, cols). Default is (m, 1)
  for m classes.

# Keyword Arguments
- `kwargs...`: Additional arguments passed to `Plots.plot()`, such as:
  - `title`: Title for the plot
  - `legend`: Legend position (default: `:none`)
  - `lw`: Line width
  - `color`: Color scheme

# Returns
A Plots.jl plot object showing the coefficient paths
"""
function Plots.plot(fit::SLOPE.SlopeFit; xvar::Symbol=:α, layout=nothing, kwargs...)
  coefs = fit.coefficients
  path_length = length(coefs)
  α = fit.α

  p, m = size(coefs[1])
  coef_matrix = zeros(p, path_length)

  for (i, coef) in enumerate(coefs)
    for row in 1:p
      val = coef[row, 1]
      coef_matrix[row, i] = val
    end
  end

  plot_defaults = Dict{Symbol,Any}(
    :xlabel => String(xvar),
    :ylabel => "β",
    :legend => :none,
  )

  if xvar == :α
    plot_defaults[:xlabel] = "α"
    plot_defaults[:xflip] = true
    plot_defaults[:xscale] = :ln
    x_values = α
  elseif xvar == :step
    plot_defaults[:xlabel] = "Step"
    x_values = collect(1:path_length)
  end

  plot_options = merge(plot_defaults, Dict(kwargs))

  # plt = Plots.plot(
  #   x_values,
  #   coef_matrix';
  #   plot_options...
  # )

  if m == 1
    coef_matrix = zeros(p, path_length)

    for (i, coef) in enumerate(coefs)
      for row in 1:p
        coef_matrix[row, i] = coef[row, 1]
      end
    end

    plt = Plots.plot(
      x_values,
      coef_matrix';
      plot_options...
    )

    return plt
  else
    plots = []

    for class in 1:m
      coef_matrix = zeros(p, path_length)

      for (i, coef) in enumerate(coefs)
        for row in 1:p
          coef_matrix[row, i] = coef[row, class]
        end
      end

      class_options = copy(plot_options)
      class_options[:title] = "Class $class"

      plt = Plots.plot(
        x_values,
        coef_matrix';
        class_options...
      )

      push!(plots, plt)
    end

    if isnothing(layout)
      layout = (m, 1)
    end

    final_plot = Plots.plot(plots..., layout=layout)
    return final_plot
  end

  return plt
end

export plot

end
