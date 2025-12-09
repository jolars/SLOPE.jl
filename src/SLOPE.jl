"""
  module SLOPE

Sorted L-One Penalized Estimation
"""
module SLOPE
using CxxWrap
using slope_jll
@wrapmodule(slope_jll.get_libslopejll_path)

function __init__()
    return @initcxx
end

include("utils.jl")
include("regweights.jl")
include("models.jl")
include("cv.jl")
include("plots.jl")

export regweights

export slope
export slopecv
export predict

export SlopeFit
export SlopeCvResult
export SlopeGridResult

end
