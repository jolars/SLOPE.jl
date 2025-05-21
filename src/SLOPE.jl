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

include("models.jl")

export slope
export SlopeFit

if !isdefined(Base, :get_extension)
  include("../ext/PlotSLOPE/PlotSLOPE.jl")
end

end

