# Changelog

## 1.0.0 (2025-05-28)

### Features

* add `plot()` method for `SlopeFit` objects ([679ca8b](https://github.com/jolars/SLOPE.jl/commit/679ca8b31f872bd8202668a677e7893de62637ff))
* add `slope()` for fitting SLOPE models ([03a1e8b](https://github.com/jolars/SLOPE.jl/commit/03a1e8bd4cc82eb2544131b66fd0ebf531768bb7))
* enable cross-validation ([d671695](https://github.com/jolars/SLOPE.jl/commit/d6716950777429ac73f836fb415660c50af50cb9))
* make metric a symbol ([8c72d41](https://github.com/jolars/SLOPE.jl/commit/8c72d419872f0930331bcdd499b7db6456d628c6))
* return `SlopeFit` struct from `slope()` ([40d67ba](https://github.com/jolars/SLOPE.jl/commit/40d67ba2bef76698efeaad5924d26ab9f476226d))
* support scalar alphas ([6651c8a](https://github.com/jolars/SLOPE.jl/commit/6651c8adcc2462b8e5a326c2f140ff8ad0810bb9))
* use symbol for loss ([5718e20](https://github.com/jolars/SLOPE.jl/commit/5718e207eb0b801cdb6d6fdfb539627fdeef5ee1))
* use symbols for centering and scaling ([88a3035](https://github.com/jolars/SLOPE.jl/commit/88a3035fb92aeec61285d0b50e49fa61f6497946))

### Bug Fixes

* add LinearAlgebra dependency ([585f84d](https://github.com/jolars/SLOPE.jl/commit/585f84de48f70933721294bdc6e6ff2ee3d993cb))
* correctly define m dimension for multinomial ([f8dee13](https://github.com/jolars/SLOPE.jl/commit/f8dee1348c19679abba115662e2be13540c3ff0c))
* correctly handle multinomial responses ([72e0848](https://github.com/jolars/SLOPE.jl/commit/72e0848991deed6f4e0bb3d360dc413a057c5391))
* **deps:** lower linear algebra dependency to 1.10.9 ([5b4dbd7](https://github.com/jolars/SLOPE.jl/commit/5b4dbd7ca307c607629e10a819afdd870f93a46c))
* fix missing n and p in test ([92b0910](https://github.com/jolars/SLOPE.jl/commit/92b0910667a5f32dbb54d7ec384359d0dcdb07f0))
* use correct lambda sequence in multinomial test ([68fb649](https://github.com/jolars/SLOPE.jl/commit/68fb649d7c302a7da5578d34696bd3e777c9207c))
