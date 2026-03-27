# Tutorial

## Introduction

SLOPE (Sorted L-One Penalized Estimation) is a type of sparse regression problem
that uses a sorted L1-norm penalty to induce both sparsity and clustering in the
coefficients of a generalized linear model. The latter differentiates SLOPE from
other sparse regression techniques like the lasso.

SLOPE minimizes the following objective function:

```math
\frac{1}{n} \sum_{i=1}^n f(y_i, x_i^\intercal \beta) + \alpha \sum_{j=1}^p \lambda_j |\beta_{(j)}|.
```

where $f(y,\eta)$ is the negative log-likelihood contribution of a single
observation $(y, \eta)$. $\beta_{(j)}$ is the $j$-th coefficient, $\beta_0$ is
the intercept, and $\lambda$ is a decreasing sequence of regularization weights.
$x_i$ is the $i$th row of the design matrix, and $n$ is the number of
observations.

## Fitting Models

The main entry point for fitting SLOPE models is the `slope()` function. It
takes as input a design matrix `X` and a response vector (or matrix) `y`.

In the following example, we fit a SLOPE model to the Boston housing dataset
from the RDatasets package, predicting the crime rate (`Crim`) using all other
features.

```@example boston
using SLOPE
using Plots
using RDatasets

boston = dataset("MASS", "Boston")

x = Matrix(boston[:, Not(:Crim)])
y = boston[:, :Crim]

fit = slope(x, y)
```

SLOPE features plotting recipes for visualizing the coefficient paths, so you
simply call `plot()` on the fitted model after having loaded the Plots package.

```@example boston
plot(fit)
savefig("plot.svg"); nothing # hide
```

![](plot.svg)

There are also several convenience functions for working with the fitted model,
such as `coef()` for extracting coefficients at specific regularization levels.

```@example boston
coef(fit, index=8)
```

## Cross-validation workflow

You can use cross-validation to select tuning parameters and then retrieve a
final model fitted on the full dataset at the selected setting.

```@example boston
cvfit = slopecv(x, y, q=[0.05, 0.1], γ=[0.0], n_folds=5)

# Selected hyperparameters and alpha
cvfit.best_params
best_α(cvfit)

# Final model at selected setting
fit_cv = best_model(cvfit)

# Equivalent explicit refit
fit_cv2 = refit(cvfit, x = x, y = y)
```

`refit` always requires explicit `x` and `y`.
