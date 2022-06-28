
# depends on:
# `linear_constraints.jl` for type promotion of `VectorAffineFunction`s

#=======================================================================
LinearConstraintsCache
========================================================================#
struct LinearConstraintsCache{
	T <: Number,
	N
}

	scaled_lower_bounds_vector :: SVector{T, N}
	scaled_upper_bounds_vector :: SVector{T, N}

	transformed_linear_equality_constraints :: Dictionary{
		ConstraintIndexEq,
		MOI.VectorAffineFunction{T}
	}
	transformed_linear_inequality_constraints :: Dictionary{
		ConstraintIndexIneq,
		MOI.VectorAffineFunction{T}
	}
end

Base.broadcastable( lcc :: LinearConstraintsCache ) = Ref( lcc )

function LinearConstraintsCache( 
	mop :: AbstractMOP,
	scal :: AbstractAffineScaler
)
	lb, ub = _bounds_vectors(mop)
	scaled_lb = transform(lb, scal)
	scaled_ub = transform(ub, scal)

	n_vars = length(lb)

	T1 = Base.promote_eltype(lb, ub)

	transformed_linear_equality_constraints = reduce(
		merge,	# should convert the value type appropriately
		Dictionary(
			ind, 
			_get(mop, ind) for ind = _eq_constraint_indices(mop)
		);
		init = Dictionary{ConstraintIndexEq, MOI.VectorAffineFunction{T1}}()
	)

	T2 = Base.promote_type( 
		T1, 
		_precision(valtype(transformed_linear_equality_constraints)) 
	)

	transformed_linear_inequality_constraints = reduce(
		merge,	# should convert the value type appropriately
		Dictionary(
			ind, 
			_get(mop, ind) for ind = _eq_constraint_indices(mop)
		);
		init = Dictionary{ConstraintIndexIneq, MOI.VectorAffineFunction{T2}}()
	)

	T = Base.promote_type(
		T2,
		_precision(valtype(transformed_linear_equality_constraints))
	)
	
	return LinearConstraintsCache(
		SVector{n_vars,T}(scaled_lb),
		SVector{n_vars,T}(scaled_ub),
		convert(
			Dictionary{ConstraintIndexEq,T}, 
			transformed_linear_equality_constraints
		),
		convert(
			Dictionary{ConstraintIndexInEq,T},
			transformed_linear_inequality_constraints
		),
	)
end

#=======================================================================
Iterate
========================================================================#
@with_kw struct Iterate{F<:Real, VT <: AbstractVector{F}}
	unscaled_vector :: VT
	scaled_vector :: VT

	unscaled_dict :: Dictionary{VariableIndex, F}
	scaled_dict :: Dictionary{VariableIndex, F}
end

iter_site_vector(iterate :: Iterate) = iterate.unscaled_vector
scaled_iter_site_vector(iterate :: Iterate) = iterate.scaled_vector
iter_site_dict(iterate :: Iterate) = iterate.unscaled_dict
scaled_iter_site_dict(iterate :: Iterate) = iterate.scaled_dict

_precision( :: Iterate{F,VT} ) where{F,VT} = F

#=======================================================================
IterData
========================================================================#

@with_kw struct IterData{
	F <: Real,
	VT <: AbstractVector{F}
}
	iterate :: Iterate{F,VT}

	evaluation_cache :: Dictionary{FunctionIndex, F}

	radius :: F 

	x_index :: Int

	objective_indices :: Indices{ObjectiveIndex}
	eq_constraint_indices :: Indices{ConstraintIndexEq}
	ineq_constraint_indices :: Indices{ConstraintIndexIneq}
	nl_eq_constraint_indices :: Indices{NLConstraintIndexEq}
	nl_ineq_constraint_indices :: Indices{NLConstraintIndexIneq}
end

_precision( :: IterData{F,VT} ) where{F,VT} = F

Base.broadcastable( id :: IterData ) = Ref( id )

@forward IterData.iterate iter_site_vector, scaled_iter_site_dict, iter_site_dict, scaled_iter_site_vector
