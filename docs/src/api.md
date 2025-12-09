# API Reference

## Modules

SLOPE only contains a single module:

```@docs
SLOPE.SLOPE
```

## Model Fitting

The main entry point for fitting SLOPE models is the
`slope()` function, which returns a `SlopeFit` object.

```@docs
SLOPE.slope
SLOPE.SlopeFit
```

## Cross-Validation

SLOPE supports native cross-validation via the `slopecv()` function,
which returns a `SlopeCvResult` object that
in turn contains a `SlopeGridResult` object

```@docs
SLOPE.slopecv
SLOPE.SlopeGridResult
SLOPE.SlopeCvResult
```

## Plotting

SLOPE provides recipes for plotting coefficient
paths as well as cross-validation results.

```@autodocs
Modules = [SLOPE]
Pages = ["plots.jl"]
```

## Utilities

For convenience, we also provide a utility function to generate
regularization weights for SLOPE.

```@docs
SLOPE.regweights
```
