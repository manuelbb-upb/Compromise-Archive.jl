
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
