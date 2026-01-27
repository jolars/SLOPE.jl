# Copilot Instructions for SLOPE.jl

## Project Overview

SLOPE.jl is a Julia package implementing Sorted L-One Penalized Estimation (SLOPE) for generalized linear models with sorted L1-norm regularization. The package is a thin wrapper around [libslope](https://github.com/jolars/libslope), a C++ library for fitting SLOPE models using CxxWrap.jl for Julia bindings.

**Repository Size**: Small (~30 source files)  
**Primary Language**: Julia  
**Key Dependencies**: CxxWrap.jl (C++ wrapper), slope_jll (binary library), RecipesBase (plotting)  
**Julia Version**: 1.7+ (tested on 1.10, 1.11, 1.12)  
**Project Type**: Statistical machine learning library

## Project Structure

```
SLOPE.jl/
├── src/              # Source code
│   ├── SLOPE.jl      # Main module file
│   ├── models.jl     # Core model implementations
│   ├── cv.jl         # Cross-validation
│   ├── coef.jl       # Coefficient extraction
│   ├── plots.jl      # Plotting recipes
│   └── utils.jl      # Utilities
├── test/             # Test suite
│   ├── runtests.jl   # Main test file
│   ├── quadratic.jl  # Tests for quadratic loss
│   ├── logistic.jl   # Tests for logistic regression
│   ├── multinomial.jl # Tests for multinomial regression
│   ├── coef.jl       # Coefficient extraction tests
│   ├── plots.jl      # Plotting tests
│   ├── cv.jl         # Cross-validation tests
│   └── regweights.jl # Regularization weight tests
├── docs/             # Documentation
│   ├── make.jl       # Documenter build script
│   ├── Project.toml  # Docs dependencies
│   └── src/          # Documentation sources
├── Project.toml      # Package dependencies
├── Taskfile.yml      # Task automation (replaces Makefile)
├── flake.nix         # Nix development environment
└── .github/workflows/ # CI/CD pipelines
```

## Build and Test Instructions

### Environment Setup

#### Option 1: Standard Julia (CI, most users)

**CRITICAL**: Always run `julia --project=.` to use the local project environment. The package requires Julia 1.7 or later.

```bash
# Install dependencies (takes ~2-3 minutes on first run)
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

#### Option 2: Nix Development Environment (Local development)

If working locally with the Nix development environment:

```bash
# Enter nix shell (provides Julia, Quarto, and other tools)
nix develop

# Then run Julia commands as normal
julia --project=. -e 'using Pkg; Pkg.instantiate()'
```

**Note**: The nix environment is optional and only needed if you don't have Julia installed system-wide. CI uses standard Julia without Nix.

### Installation and Build

**Expected behavior**: Downloads and compiles dependencies including CxxWrap (~60 seconds), slope_jll, and test dependencies (Plots). Total precompilation time: 60-90 seconds.

**Common issues**: 
- Network errors downloading Julia registry are non-fatal warnings
- CxxWrap compilation is the longest step (50-60 seconds)

### Running Tests

**ALWAYS run tests before finalizing code changes**:
```bash
julia --project=. -e 'using Pkg; Pkg.test()'
```

**Expected time**: ~25 seconds total (includes Aqua quality checks: ~14s)

**Test structure**: Tests are organized by model type:
- Quadratic (ordinary least squares)
- Logistic (binary classification)
- Multinomial (multi-class)
- Coefficients (extraction and interpolation)
- Plots (visualization)
- CV (cross-validation)
- Regularization weights
- Aqua (code quality checks)

**All tests must pass** before submitting changes. Expected: 81 passing tests.

### Alternative Build Methods

The `Taskfile.yml` provides convenience targets:
```bash
task install  # Same as Pkg.instantiate()
task test     # Same as Pkg.test()
task docs     # Build documentation
task readme   # Regenerate README (requires Quarto)
```

### Documentation Building

Build documentation locally:
```bash
julia --project=docs/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
julia --project=docs/ docs/make.jl
```

**Time**: 1-2 minutes. Output in `docs/build/`.

### Quarto Documentation

The package supports Quarto for generating tutorial-style documentation:

1. Create `.qmd` files in `docs/src/`
2. Render to markdown: `quarto render docs/src/tutorial.qmd` (requires IJulia kernel for code execution)
3. Add rendered `.md` to `docs/make.jl`

**Note**: Code execution in Quarto requires IJulia. For now, use `execute: enabled: false` in YAML header or write docs without code execution.

## CI/CD Workflows

### Main Workflow: build-and-test.yml
Triggered on: push to main, PRs, tags, manual dispatch

Tests run on:
- Ubuntu latest
- Julia versions: 1.10, 1.11, 1.12
- Architecture: x64 only
- Timeout: 60 minutes

**CRITICAL**: If you add code using new Julia features, verify compatibility with Julia 1.7+ (specified in Project.toml).

### Documentation: documenter.yml
Builds and deploys documentation on push to main and tags.

### Other Workflows
- **compat-helper.yml**: Automatic dependency updates (daily)
- **tagbot.yml**: Automatic release tagging
- **dependabot**: Weekly GitHub Actions updates

## Key Development Guidelines

### Code Contributions

**IMPORTANT**: Most algorithmic changes should be made in the [libslope C++ library](https://github.com/jolars/libslope), not in this wrapper. This package primarily provides:
1. Julia bindings via CxxWrap
2. High-level API (`slope()`, `slopecv()`, `predict()`, `coef()`)
3. Plotting recipes
4. Cross-validation utilities
5. Pretty printing for results

### Commit Message Format

**ALWAYS use conventional commits format**:
```
type(scope): description

Examples:
feat(cv): add k-fold cross-validation
fix(models): correct lambda sequence generation
docs: update API documentation
test: add tests for multinomial model
```

Types: feat, fix, docs, test, refactor, style, chore, perf, ci

### File Modification Guidelines

**Key configuration files** (modify with care):
- `Project.toml`: Package metadata and dependencies (Aqua is in [extras] only, not [deps])
- `src/SLOPE.jl`: Module definition and CxxWrap initialization
- `src/models.jl`: Core model fitting logic and `SlopeFit` struct with `Base.show` methods
- `src/coef.jl`: Coefficient extraction with alpha interpolation
- `src/cv.jl`: Cross-validation with `SlopeCvResult` struct and `Base.show` methods
- `test/runtests.jl`: Test suite organization

**DO NOT modify**:
- `Manifest.toml` (gitignored, auto-generated)
- `README.md` (generated from readme/README.qmd)
- Binary artifacts in `.julia/artifacts/`

### Testing Requirements

**Before submitting**:
1. Run full test suite: `julia --project=. -e 'using Pkg; Pkg.test()'`
2. Ensure all 81 tests pass (including Aqua.jl code quality checks)
3. Test on Julia 1.7+ if possible (CI tests 1.10, 1.11, 1.12)
4. Verify no regressions in existing functionality

### Code Style

- Follow Julia standard style conventions
- Use 2-space indentation (Julia convention)
- Add docstrings for exported functions
- Keep functions focused and concise
- Use Unicode symbols where appropriate (e.g., `α` instead of `alpha`)
- Comments are minimal in existing code; match existing style
- Custom `Base.show` methods provide user-friendly display of results

## Dependencies and Compatibility

### Core Dependencies
- **CxxWrap.jl** (0.17.0): C++ bindings - version pinned
- **slope_jll** (6.3): Binary library - version pinned
- **RecipesBase** (1.3.4): Plotting integration
- **Distributions** (0.25.122): Statistical distributions
- **StatsBase** (0.34.9): Statistical utilities
- Standard library: LinearAlgebra, Random, SparseArrays

### Test Dependencies
- **Aqua** (0.8): Code quality tests (in [extras] only)
- **Plots** (1.40.13): Plotting tests
- **Test**: Standard library

**Compatibility constraints in Project.toml**: Julia 1.7+, specific versions for key packages. Do not relax constraints without thorough testing.

## Common Issues and Solutions

### Issue: CxxWrap precompilation is slow
**Expected behavior**: First precompilation takes 50-60 seconds. Subsequent runs use cache.

### Issue: Tests timeout
**Solution**: Default timeout is 60 minutes. Tests normally complete in ~25 seconds. If timeout occurs, check for infinite loops or network issues.

### Issue: Binary library not found
**Solution**: Run `Pkg.instantiate()` to download slope_jll artifacts. Check that Julia can access GitHub for artifact downloads.

### Issue: README changes not reflected
**Solution**: README.md is generated from readme/README.qmd using Quarto. Edit the .qmd file, then run:
```bash
julia --project=readme/ -e 'using Pkg; Pkg.develop(PackageSpec(path=pwd())); Pkg.instantiate()'
quarto render readme/README.qmd --output README.md
mv readme/README_files README_files
```

### Issue: Documentation build fails
**Solution**: Ensure you're using the docs project environment:
```bash
julia --project=docs/ docs/make.jl
```

### Issue: Nix environment needed locally
**Solution**: If Julia is not available system-wide, use `nix develop` to enter the development shell which provides Julia, Quarto, and build tools.

## Quick Reference

### Essential Commands
```bash
# Setup (first time)
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Run tests (always before PR)
julia --project=. -e 'using Pkg; Pkg.test()'

# Build docs
julia --project=docs/ docs/make.jl

# With nix (local development)
nix develop
julia --project=. -e 'using Pkg; Pkg.instantiate()'

# Using task runner
task install
task test
task docs
```

### File Locations Reference
- Main module: `src/SLOPE.jl`
- Model API: `src/models.jl` (includes `Base.show` for `SlopeFit`)
- Cross-validation: `src/cv.jl` (includes `Base.show` for `SlopeCvResult`)
- Coefficients: `src/coef.jl` (includes `α` parameter with interpolation)
- Plotting: `src/plots.jl`
- Test entry: `test/runtests.jl`
- CI config: `.github/workflows/build-and-test.yml`

## Trust These Instructions

These instructions have been validated by:
1. Successfully installing dependencies from a clean environment
2. Running the complete test suite (81 tests pass)
3. Building documentation
4. Examining all workflow configurations
5. Testing both standard Julia and Nix development environments

**If something works differently than described here, the instructions may be outdated**. Verify against the actual files and update this document accordingly.
