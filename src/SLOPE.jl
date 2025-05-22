"""
  module SLOPE

Sorted L-One Penalized Estimation
"""
module SLOPE
using CxxWrap
using slope_jll
@wrapmodule(slope_jll.get_libslopejll_path)

function __init__()
  @initcxx
end

include("utils.jl")
include("models.jl")
include("cv.jl")

export slope
export slopecv

export SlopeFit
export SlopeCvResult
export SlopeGridResult

# Make extensions backward compatible with Julia < 1.9
if !isdefined(Base, :get_extension)
  include("../ext/PlotSLOPE/PlotSLOPE.jl")
end

end

