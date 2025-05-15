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

$$
\frac{1}{n} \sum_{i=1}^n f(y_i, x_i^\intercal \beta) + \sum_{j=1}^p \lambda_j |\beta_{(j)}|,
$$

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

```

```
