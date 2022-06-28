# Imports required:
# * Dictionaries
# * MathOptInterface as MOI
# * SparseArrays: sparce
# Definitions required:
# * `VariableIndex`
# * `AbstractAffineScaler`
# * `AbstractMOP`

# ## IMPORTANT 
# Througouht this file we always assume linear constraints to be of the form 
# ``Ax + b ≦ 0`` and `MOI.VectorAffineFunction`s to be such that they 
# store ``b`` correspondingly, without a change of the sign.

_precision( :: Type{<: MOI.VectorAffineFunction{T}} ) where T = T
_precision( :: T ) where T <: MOI.VectorAffineFunction = _precision(T)

# convert a VectorAffineFunction with precision `F` to one of precision `T`
function Base.convert(
	::Type{<:MOI.VectorAffineFunction{T}}, 
	vaf :: MOI.VectorAffineFunction{F}
) where{T<:Number, F<:Number}
	return MOI.VectorAffineFunction{T}(
		MOI.VectorAffineTerm{T}[ 
			MOI.VectorAffineTerm(
				t.output_index,
				MOI.ScalarAffineTerm{T}(
					convert(T, t.scalar_term.coefficient),
					t.scalar_term.variable
				)
			)
		for t in vaf.terms],
		T.(vaf.constants)
	)
end

"Transform a `aff_func::MOI.ScalarAffineFunction` to a `MOI.VectorAffineFunction`."
function _scalar_to_vector_aff_func( aff_func :: MOI.ScalarAffineFunction )
    vec_terms = [ MOI.VectorAffineTerm( 1, term) for term = aff_func.terms ]
    consts = [ aff_func.constant, ]
    return MOI.VectorAffineFunction( vec_terms, consts )
end

# evaluation of linear constraints
function _eval_vaf( 
    vaf :: MOI.VectorAffineFunction{T}, 
    xd :: Union{AbstractDict{MOI.VariableIndex,R},AbstractDictionary{MOI.VariableIndex, R}}
) where {T,R}
	num_out = MOI.output_dimension(vaf)
	F = Base.promote_type( MIN_PRECISION, Base.promote_op( *, T, R ) )
	ret = convert( MVector{num_out, F}, vaf.constants )
	for vaf_term in vaf.terms
		s_term = vaf_term.scalar_term
		ret[ vaf_term.output_index ] +=	s_term.coefficient * getindex(xd, s_term.variable);
	end
	return ret
end

function moi_variable_indices( vaf )
    vars = MOI.VariableIndex[]
    for t in vaf.terms
        st = t.scalar_term
        st_v = st.variable 
        if !(st_v in vars)
            push!(vars, st_v)
        end
    end
    return vars
end
#=
# only defined used for compatibility with the inner evaluator interface
# should not be used anywhere
function _eval_vaf_at_vec(vaf::MOI.VectorAffineFunction, x::Vec)
    xd = Dictionary(
        moi_variable_indices(vaf),
        x
    )
    return _eval_vaf(vaf, xd)
end
=#

"""
	_arrays_to_vector_affine_function(
		A, b = zeros(eltype(A), size(A,1)); 
		variables 
	)

Provided a ``m×n`` matrix `A` and a vector `b`, describing a constraint 
``Ax + b ≦ 0``, return a ``MOI.VectorAffineFunction`` to use as 
a constraint for JuMP models with ``MOI.Nonpositives``.
"""
function _arrays_to_vector_affine_function( 
	A :: AbstractMatrix{F}, 
	b :: AbstractVector{T} = zeros(F, size(A,1)); 
	variables
) where{F<:Number, T<:Number}
	m, n = size(A)
	@assert n == length(vars) "`A` must have the same number of columns as there are `vars`."
	@assert m == length(b) "`A` must have the same number of rows as there are entries in `b`."

	S = Base.promote_type(F, T)
	terms = collect(Iterators.flatten(
		[ 	
			[ 	
				MOI.VectorAffineTerm( 
					i, 
					MOI.ScalarAffineTerm( S(row[j]), variables[j] ) 
				) 
			for j = 1:n ] 
		for (i,row) = enumerate( eachrow(A) ) ] 
	))
	constants = S.(b)
	return MOI.VectorAffineFunction( terms, constants )
end

function _vector_affine_function_to_matrix(
	vaf :: MOI.VectorAffineFunction{T};
	variables_columns_mapping :: AbstractDictionary{MOI.VariableIndex,Int} = Dictionary{MOI.VariableIndex,Int}()
) where T
	num_terms = length(vaf.terms)
	row_inds = Vector{Int}(undef, num_terms)
	col_inds = Vector{Int}(undef, num_terms)
	vals = Vector{T}(undef, num_terms)
	for (i,term) in enumerate(vaf.terms)
		s_term = term.scalar_term
		row_inds[i] = term.output_index
		col_inds[i] = get!( 
			variables_columns_mapping, 
			s_term.variable,
			length(variables_columns_mapping) + 1
		)
		vals[i] = s_term.coefficient
	end
	return sparse( 
		row_inds, 
		col_inds, 
		vals,
		MOI.output_dimension(vaf),	# m =̂ number of rows
		length(variables_columns_mapping),	# n =̂ number of colums
		+ # combine function for duplicate entries
	)
end 

function _vector_affine_function_to_vector(
	vaf :: MOI.VectorAffineFunction{T};
) where T
	return vaf.constants
end 

function _vector_affine_functions_to_matrix(
	vafs :: AbstractVector{<:MOI.VectorAffineFunction};
	variables_columns_mapping :: AbstractDictionary{MOI.VariableIndex,Int}
)
	return reduce(
		vcat,
		_vector_affine_function_to_matrix.(
			vafs;
			variables_columns_mapping
		)
	)
end

function _vector_affine_functions_to_vector(
	vafs :: AbstractVector{<:MOI.VectorAffineFunction};
)
	return reduce(
		vcat,
		_vector_affine_function_to_matrix.(vafs)
	)
end

"""
	transform_vector_affine_function(vaf, scal)

Return a `MOI.VectorAffineFunction` that is applicable in the 
scaled domain given by `scal::AbstractAffineScaler`.
More precisely, suppose that `vaf` describes a constraint 
``Ax + b ≦ 0`` and that `scal` applies the transformation 
``ξ = Dx + c``.
Then the new `MOI.VectorAffineFunction` should match 
```math
A(D^{-1}(ξ - c)) + b ≦ 0 :⇔ Ã ξ + b̃ ≦ 0,
```
with ``Ã = AD^{-1}`` and ``b̃ = b - AD^{-1} c``.
"""
function transform_vector_affine_function(
	vaf :: MOI.VectorAffineFunction{T},
	scal :: AbstractAffineScaler
) where T
	scaler_vars = _variable_indices(scal)
	A = _vector_affine_function_to_matrix( 
		vaf;
		variables_columns_mapping = Dictionary( 
			scaler_vars, eachindex( scaler_vars )
		)
	)
	D_inv = _unscaling_matrix( scal )
	c = _scaling_constants_vector( scal )
	A_tilde = A*D_inv
	
	return _arrays_to_vector_affine_function(
		A_tilde, vaf.constants .- A_tilde * c;
		variables = scaler_vars
	)
end



#=
@witk_kw struct VAFInnerEvaluator{
    R<:Number, 
    AT<:AbstractMatrix{R}, 
    BT<:AbstractVector{R}
}<: AbstractInnerEvaluator
    vaf :: MOI.VectorAffineFunction{R}
    num_outputs :: MOI.output_dimension(vaf)
    counter :: Base.RefValue{Int} = Ref(0)

    moi_variables :: Vector{MOI.VariableIndex} = moi_variable_indices(vaf)
    n_vars :: Int = length(moi_variable_indices)
    A :: AT = let _A = _vector_affine_function_to_matrix(vaf);
        density = sum(iszero.(_A)) / length(_A)
        if density <= .3
            sparse(_A)
        else
            if n_vars * num_outputs < 100
                SMatrix{num_outputs, n_vars}(_A)
            else
                _A
            end
        end
    end

    b :: BT = let _b = _vector_affine_function_to_vector(vaf);
        density = sum(iszero.(_b)) / length(_b)
        if density <= .3
            sparse(_b)
        else
            if n_vars < 100
                SVector{n_vars}(_b)
            else
                _b
            end
        end
    end
end

num_eval_counter(vie::VAFInnerEvaluator) = vie.counter
num_outputs(vie::VAFInnerEvaluator) = vie.num_outputs

function _eval_at_vec(vie::VAFInnerEvaluator, x::Vec)
    return vie.A * x .+ vie.b
end

function _eval_at_vecs(vie::VAFInnerEvaluator, X::VecVec)
    _X = reduce(hcat, X)    # can this be done lazily? ApplyArray(hcat, X...), but splatting is slow
    _tmp = vie.A * _X .+ vie.b
    return collect(eachcol(_tmp))
end

_provides_gradients(::VAFInnerEvaluator) = true
_provides_jacobian(::VAFInnerEvaluator) = true
_provides_hessian(::VAFInnerEvaluator) = true

function _gradient(vie::VAFInnerEvaluator, x::Vec; output_number)
    return vec(vie.A[output_number,:])
end
function _jacobian(vie::VAFInnerEvaluator, x::Vec)
    return vie.A
end
function _partial_jacobian(vie::VAFInnerEvaluator, ::Vec; output_numbers)
    return vie.A[output_numbers, :]
end
function _hessian(vie::VAFInnerEvaluator,::Vec; output_number)
    return spzeros(vie.n_vars, vie.n_vars)
end
    
@with_kw struct VAFOuterEvaluator{
    R<:Number, 
    II <: AbstractIndices{VariableIndex}
    IT <: VAFInnerEvaluator{R}
} <: AbstractOuterEvaluator
    inner_evaluator :: IT
    
    input_indices :: II
    num_inputs = length(input_indices)
    
    @assert num_inputs == inner_evaluator.n_vars
end
=#