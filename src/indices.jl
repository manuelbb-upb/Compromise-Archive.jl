abstract type AbstractIndex end
Base.broadcastable(i::AbstractIndex) = Ref(i)

abstract type ScalarIndex <: AbstractIndex end
abstract type FunctionIndex <: AbstractIndex end
abstract type ConstraintIndex <: FunctionIndex end

"Index type to reference scaler variables. Internally, simply wraps an integer."
struct VariableIndex <: ScalarIndex
    value::Int
    name::String

    #moi_var :: MOI.VariableIndex
    function VariableIndex(value::Int, name::String="")
        new(
            value,
            name,
            #MOI.VariableIndex(value)
        )
    end
end

#const VariableIndex = MOI.VariableIndex

Base.@kwdef struct DependentIndex <: ScalarIndex
    uid::UUIDs.UUID = UUIDs.uuid4()
end

struct ScaledVariableIndex <: ScalarIndex
    inner :: VariableIndex
end

#const ScalarIndex = Union{VariableIndex, DependentIndex}

"""
	ObjectiveIndex( value::Int, name="" )

Index to reference a vector-valued objective function in an `AbstractMOP`
and store the output dimension.
"""
struct ObjectiveIndex <: FunctionIndex
    value::Int
    name::String

    ObjectiveIndex(val::Int, name::String="") = new(val, name)
end

for tn = [:NLConstraintIndexEq, :NLConstraintIndexIneq, :ConstraintIndexEq, :ConstraintIndexIneq]
    @eval begin
        """
          $($(tn))( value::Int, name="" )

        Index to reference a vector-valued constraint function
        of type `AbstractOuterEvaluator` in an `AbstractMOP`
        and store the output dimension (for convenience).
        """
        struct $(tn) <: ConstraintIndex
            value::Int
            name::String

            function $(tn)(val::Int, name::String="")
                new(val, name)
            end
        end
    end
end

#=ConstraintIndex = Union{
    NLConstraintIndexEq,
    NLConstraintIndexIneq,
    ConstraintIndexEq,
    ConstraintIndexIneq,
}=#

#FunctionIndex = Union{ObjectiveIndex, ConstraintIndex}

FunctionIndexTuple = Tuple{Vararg{<:FunctionIndex}}
FunctionIndexIterable = Union{FunctionIndexTuple,AbstractVector{<:FunctionIndex}}

@with_kw struct InnerIndex <: AbstractIndex
    uid::UUIDs.UUID = UUIDs.uuid4()
end

function concrete_subtypes(T)
    ret_types = []
    for S in subtypes(T)
        if isconcretetype(S)
            push!(ret_types, S)
        else
            append!(ret_types, concrete_subtypes(S))
        end
    end
    return ret_types
end

const FUNCTION_INDEX = Union{concrete_subtypes(FunctionIndex)...}
const SCALAR_INDEX = Union{concrete_subtypes(ScalarIndex)...}
