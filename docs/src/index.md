# SLOPE

## Installation

You can install the package using the Julia package manager:

```julia
]add SLOPE
```

Alternatively, you can also install the latest development version of the
package from the source code on GitHub by calling

```julia
using Pkg
Pkg.add(url = "https://github.com/jolars/SLOPE.jl")
```

## Getting Started

SLOPE is a Julia package for Sorted L1 Penalized Estimation (SLOPE), which
is a type of regularized regression. SLOPE minimizes the following
objective function:

```math
\frac{1}{n} \sum_{i=1}^n f(y_i, x_i^\intercal \beta) + \alpha \sum_{j=1}^p \lambda_j |\beta_{(j)}|.
```

where $f(y,\eta)$ is the negative log-likelihood contribution of
a single observation $(y, \eta)$. $\beta_{(j)}$ is the $j$-th
coefficient, $\beta_0$ is the intercept, and $\lambda$ is a
decreasing sequence of regularization weights. $x_i$ is the
$i$th row of the design matrix, and $n$ is the number of
observations.

SLOPE is a type of sparse regression, which means that
it will, given high enough penalization, set some of the
coefficients to zero, effectively removing the
corresponding features from the model. If you are familiar
with the lasso, then you should know that SLOPE is
actually a generalization of the lasso (which you can
see by setting all $\lambda_j$ to the same value).
Unlike the lasso, however, SLOPE also clusters
coefficients by settings them to the same magnitude.
This helps remove some deficiencies of the lasso
in highly correlated settings, and it also
allows for selection of more features that is
possible with the lasso.

### Basic Usage

First, we'll load the package and fit a simple model:

```julia
using SLOPE
using Random
using Statistics

# Generate some sample data
n, p = 100, 20
Random.seed!(123)
X = randn(n, p)
β = vcat(fill(3.0, 5), fill(0.0, p-5))  # 5 non-zero coefficients
y = X * β + 0.5 * randn(n)

# Fit a SLOPE model
fit = slope(X, y)
```

### Examining the Model

You can examine the results of the fitted model:

```julia
fit.coefficients
```

### Cross-Validation

To determine the optimal regularization strength, you can use cross-validation:

```julia
# Perform cross-validation to find optimal parameters
cvfit = slopecv(X, y)

# View the optimal α value
cvfit.best_params
```

## Contributing

The SLOPE.jl package is a thin wrapper around the [C++ slope
library](https://github.com/jolars/libslope), which provides all of the core
functionality. Therefore, if you find any bugs or have feature requests, then
it's likely that you should open a ticket in the slope repository rather than
here.

That being said, if you find any bugs in the Julia wrapper
or there are features in the C++ library that are yet to
be implemented in the Julia wrapper, then please open an issue
in this repository.
