abstract type AbstractIndex end
Base.broadcastable( i :: AbstractIndex ) = Ref(i)

#=
"Index type to reference scaler variables. Internally, simply wraps an integer."
struct VariableIndex <: AbstractIndex
	value :: Int 
	name :: String
	
	VariableIndex( value :: Int, name :: String = "") = new(value, name)
end
=#

const VariableIndex = MOI.VariableIndex

@with_kw struct DependentIndex <: AbstractIndex
	uid :: UUIDs.UUID = UUIDs.uuid4()
end

const ScalarIndex = Union{VariableIndex, DependentIndex}

"""
	ObjectiveIndex( value::Int, num_out::Int )

Index to reference a vector-valued objective function in an `AbstractMOP`
and store the output dimension.
"""
struct ObjectiveIndex <: AbstractIndex
    value :: Int
    name :: String 

    ObjectiveIndex( val :: Int, name :: String = "" ) = new(val, name)
end

for tn = [:NLConstraintIndexEq, :NLConstraintIndexIneq, :ConstraintIndexEq, :ConstraintIndexIneq]
    @eval begin
		"""
			$($(tn))( value::Int, num_out::Int )
		
		Index to reference a vector-valued constraint function 
		of type `AbstractOuterEvaluator` in an `AbstractMOP`
		and store the output dimension (for convenience).
		"""
		struct $(tn) <: AbstractIndex
			value :: Int
			name :: String 
			
			function $(tn)( val :: Int, name :: String = "" )
				new(val, name)
			end
		end
	end
end

ConstraintIndex = Union{
    NLConstraintIndexEq,
    NLConstraintIndexIneq,
    ConstraintIndexEq,
    ConstraintIndexIneq,
}

FunctionIndex = Union{ObjectiveIndex, ConstraintIndex}

FunctionIndexTuple = Tuple{Vararg{<:FunctionIndex}}
FunctionIndexIterable = Union{FunctionIndexTuple, AbstractVector{<:FunctionIndex}}

@with_kw struct InnerIndex <: AbstractIndex
	uid :: UUIDs.UUID = UUIDs.uuid4()
end