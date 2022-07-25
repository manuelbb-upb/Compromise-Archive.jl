module Compromise

using StaticArrays
using Dictionaries

# used by/in:
# indices.jl
import UUIDs

import LinearAlgebra

import SparseArrays: spzeros, sparsevec, sparse, 
	AbstractSparseMatrixCSC, SparseMatrixCSC,
	AbstractSparseVector, SparseVector

import Parameters: @with_kw, @unpack, @pack! 	# convenient keyword constructors

using Preferences 	# used to set `nansafe_mode` in `ForwardDiff`
using Requires		# conditional loading of `ForwardDiff`
function __init__()
    @require ForwardDiff="f6369f11-7733-5829-9624-2563aa707210" begin 
		set_preferences!(ForwardDiff, "nansafe_mode" => true)
	end
end

import FiniteDifferences	# provided as default fallback in `_differentiation.jl`
using AbstractDifferentiation

import Lazy: @forward	# forward methods to field of wrapper type

# used by/in:
# var_scalers.jl
import InteractiveUtils: @which

# used by/in: 
# evaluators.jl
using MacroTools: isexpr

# used by/in:
# mop.jl
using InteractiveUtils: subtypes

include("constants.jl")

include("indices.jl")

include("surrogate_interface.jl")
struct DummySurrogateConfig <: AbstractSurrogateConfig end 
const DUMMY_CONFIG = DummySurrogateConfig()

include("evaluators.jl")

include("var_scalers.jl")

include("linear_constraints.jl")

include("iter_data.jl")

include("mop.jl")

include("utilities.jl")

include("algo_config.jl")

include("algorithm.jl")

end # module
