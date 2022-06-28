# Imports required:
# * Dictionaries
# * InteractiveUtils: @which
# Definitions required:
# * `ScalarIndex`
# * `AbstractAffineScaler`

Base.broadcastable( scal :: AbstractAffineScaler ) = Ref(scal)
# AbstractAffineScaler represents affine linear transformations of the form 
# x̂ = Sx + b 	# S =̂ scaling matrix, b =̂ scaling_constants
# x = S⁻¹(x̂ - b)

# mandatory
# one of 
__variables( scal :: AbstractAffineScaler ) :: Union{Nothing,AbstractVector{<:ScalarIndex}} = nothing
__variable_indices( scal :: AbstractAffineScaler ) :: Union{Nothing, Indices{<:ScalarIndex}}= nothing

__init( :: Type{<:AbstractAffineScaler}; kwargs...) = nothing 

# one of 
"To obtain the scaled value of variable `vi`, return a `Dictionary` of coefficients to apply to the unscaled variables."
__scaling_coefficients( scal :: AbstractAffineScaler, vi :: ScalarIndex) :: Union{AbstractDictionary,Nothing} = nothing
__scaling_matrix( scal :: AbstractAffineScaler) = nothing

__scaling_constant( scal :: AbstractAffineScaler, vi :: ScalarIndex) = nothing
__scaling_constants_vector( scal :: AbstractAffineScaler ) = nothing

# derived (callable) methods 
function _var_pos( scal :: AbstractAffineScaler, vi :: ScalarIndex )
    findfirst(isequal(vi), _variables_indices(scal))
end

macro not_defined_fallback( hopefully_defined_return_expr, alternative_expr )
	return quote
		hopefully_defined_return = $(esc(hopefully_defined_return_expr));
		if isnothing( hopefully_defined_return )
			return $(esc(alternative_expr))
		else
			return hopefully_defined_return
		end
	end
end

__unscaling_matrix( scal :: AbstractAffineScaler ) = LinearAlgebra.inv( _scaling_matrix(scal) )
__unscaling_coefficients( scal :: AbstractAffineScaler, vi :: ScalarIndex ) = nothing

function _variables(scal :: AbstractAffineScaler)
    vars = __variables(scal)
    isnothing(vars) && return collect( __variable_indices(scal) )
    return vars
end

function _variable_indices(scal :: AbstractAffineScaler)
    vars = __variable_indices(scal)
    isnothing(vars) && return Indices( __variables(scal) )
    return vars
end

# (helpers)
function _coeff_dict_of_dicts(scal)
	return __scaling_coefficients.(scal, _variable_indices(scal))
end

function _matrix_from_coeff_dict( vars , coeff_dict_of_dicts )
	return transpose(
		reduce( 
			hcat,
			collect(getindices( coeff_dict_of_dicts[vi], vars ) for vi = vars )
		)
	)
end

function _scaling_matrix( scal :: AbstractAffineScaler )
	return @not_defined_fallback(
		__scaling_matrix( scal ),
		_matrix_from_coeff_dict( _variable_indices(scal), _coeff_dict_of_dicts(scal) )
	)
end

function _matrix_to_coeff_dict( mat, variables, vi_pos )
	coeff_row = mat[vi_pos, :];
	dictionary( vj => coeff_row[ vj_pos ] for (vj_pos, vj) = enumerate(variables) ) 
end

function _scaling_coefficients( scal :: AbstractAffineScaler, vi :: ScalarIndex )
	return @not_defined_fallback(
		__scaling_coefficients(scal, vi),
		_matrix_to_coeff_dict( 
			__scaling_matrix( scal ), 
			_variable_indices(scal), 
			_var_pos(scal, vi) 
		)
	)
end

function _scaling_constants_vector( scal :: AbstractAffineScaler )
    return @not_defined_fallback(
		__scaling_constants_vector( scal ),
		collect(__scaling_constant.(scal, _variable_indices(scal)))
	)
end

function _scaling_constant( scal ::AbstractAffineScaler, vi :: ScalarIndex )
	return @not_defined_fallback(
		__scaling_constant(scal, vi),
		__scaling_constants_vector( scal )[ _var_pos(scal,vi) ]
	)
end

function _unscaling_coefficients( scal :: AbstractAffineScaler, vi :: ScalarIndex )
	return @not_defined_fallback(
		__unscaling_coefficients( scal, vi ),
		_matrix_to_coeff_dict( 
			__unscaling_matrix(scal), 
			_variable_indices(scal), 
			_var_pos(scal, vi) 
		)
	)
end

function _inv_coeff_dict_of_dicts(scal)
	return __unscaling_coefficients.(scal, _variable_indices(scal))
end

function _unscaling_matrix( scal :: AbstractAffineScaler )
	return @not_defined_fallback(
		__unscaling_matrix(scal),
		_matrix_from_coeff_dict( _variable_indices(scal), _inv_coeff_dict_of_dicts(scal) )
	)
end

function transform( x :: AbstractVector, scal :: AbstractAffineScaler )
	return _scaling_matrix(scal) * x .+ _scaling_constants_vector(scal)
end

function untransform( x_scaled :: AbstractVector, scal :: AbstractAffineScaler )
	return _unscaling_matrix(scal) * ( x_scaled .- _scaling_constants_vector(scal) )
end

function dot_prod( x :: AbstractDictionary, y :: AbstractDictionary )
	return sum( x .* getindices(y, keys(x)) )
end

function transform( x :: AbstractDictionary, scal :: AbstractAffineScaler )
	return map(
		vi -> dot_prod(x, _scaling_coefficients( scal, vi) ) + _scaling_constant(scal, vi),
		keys(x)
	)
end
 
function untransform( x :: AbstractDictionary, scal :: AbstractAffineScaler )
	_x = map( ((vi,xi),) -> xi - _scaling_constant(scal, vi), pairs(x) )
	return map(
		vi -> dot_prod( _x, _unscaling_coefficients(scal, vi) ),
		keys(x)
	)
end

macro set_if_needed(
	kwarg_name_expr,
	kwarg_symbols_expr,
	alternative_params_expr
)
	return quote
		local provided_params = $( esc( kwarg_name_expr ) )
		local var_val = if isnothing( provided_params ) && 
			$(Meta.quot(kwarg_name_expr)) in $(esc(kwarg_symbols_expr))
			$(esc(alternative_params_expr))
		else
			provided_params
		end
		$(esc(Symbol("🐢", kwarg_name_expr))) = var_val
	end
end

macro check_turtle_length( _arg_name, n_vars_expr )
	return quote
		arg = $(esc(_arg_name))
		if !isnothing(arg)
			len = length(arg)
			n = $(esc(n_vars_expr))
			@assert len == n "`$($(Meta.quot(_arg_name)))` has length $(len) but needs $(n)."
		end
	end
end

macro check_turtle_mat( _arg_name, n_vars_expr )
	return quote
		arg = $(esc(_arg_name))
		if !isnothing(arg)
			M, N = size(arg)
			n = $(esc(n_vars_expr))
			@assert M == n "`$($(Meta.quot(_arg_name)))` has $(M) rows but needs $(n)."
			@assert M == n "`$($(Meta.quot(_arg_name)))` has $(N) columns but needs $(n)."
		end
	end
end

function _init( scal_type :: Type{<:AbstractAffineScaler};
	variables :: AbstractVector{<:ScalarIndex} = nothing,
	variable_indices :: Union{Indices{<:ScalarIndex}, Nothing} = nothing,
	scaling_coefficients_dict_of_dicts :: Union{Nothing, AbstractDictionary{<:ScalarIndex, <:AbstractDictionary}} = nothing,
	scaling_matrix :: Union{Nothing,AbstractMatrix{<:Real}} = nothing, 
	scaling_constants_dict :: Union{Nothing, AbstractDictionary{<:ScalarIndex, <:Real}} = nothing,
	scaling_constants_vector :: Union{Nothing,AbstractVector{<:Real}} = nothing,
	unscaling_coefficients_dict_of_dicts :: Union{Nothing, AbstractDictionary{<:ScalarIndex, <:AbstractDictionary}} = nothing,
	unscaling_matrix :: Union{Nothing, AbstractMatrix} = nothing
)
	@assert (!isnothing(variables) && !isempty(variables)) || 
		(!isnothing(variable_indices) && !isempty(variable_indices))

	@assert !(isnothing(scaling_constants_dict) && isnothing(scaling_constants_vector))
	@assert any( !isnothing(_arg) for _arg in [
		scaling_coefficients_dict_of_dicts, scaling_matrix, 
		unscaling_coefficients_dict_of_dicts, unscaling_matrix
	])

	kw_arg_symbols = Base.kwarg_decl( @which( __init(scal_type) ) )
	
	var_inds = isnothing(variable_indices) ? Indices(variables) : variable_indices

	compute_unscaling_matrix = false
	if (:unscaling_matrix in kw_arg_symbols || :unscaling_coefficients_dict_of_dicts in kw_arg_symbols) && (
		isnothing(unscaling_matrix) && isnothing(unscaling_coefficients_dict_of_dicts)
	)
		compute_unscaling_matrix = true
		push!( kw_arg_symbols, :scaling_matrix )
	end

	compute_scaling_matrix = false
	if (:scaling_matrix in kw_arg_symbols || :scaling_coefficients_dict_of_dicts in kw_arg_symbols) && (
		isnothing(scaling_matrix) && isnothing(scaling_coefficients_dict_of_dicts)
	)
		compute_scaling_matrix = true
		push!( kw_arg_symbols, :unscaling_matrix )
	end

	@set_if_needed( variables, kw_arg_symbols,
		collect(variables)
	)

	@set_if_needed( scaling_constants_dict, kw_arg_symbols,
		Dictionary( var_inds, scaling_constants_vector )
	)

	@set_if_needed( scaling_constants_vector, kw_arg_symbols,
		collect( getindices( scaling_constants_dict, var_inds) )
	)

	if compute_scaling_matrix
		@set_if_needed( unscaling_matrix, kw_arg_symbols,
			_matrix_from_coeff_dict( var_inds, unscaling_coefficients_dict_of_dicts )
		)
		# 🐢unscaling_matrix is now available
		scaling_matrix = LinearAlgebra.inv(🐢unscaling_matrix)
	else
		@set_if_needed( scaling_matrix, kw_arg_symbols,
			_matrix_from_coeff_dict( var_inds, scaling_coefficients_dict_of_dicts )
		)
		scaling_matrix = 🐢scaling_matrix
	end

	if compute_unscaling_matrix
		@set_if_needed( scaling_matrix, kw_arg_symbols,
			_matrix_from_coeff_dict( var_inds, scaling_coefficients_dict_of_dicts )
		)
		# 🐢scaling_matrix is now available
		unscaling_matrix = LinearAlgebra.inv(🐢scaling_matrix)
	else
		@set_if_needed( unscaling_matrix, kw_arg_symbols,
			_matrix_from_coeff_dict( var_inds, unscaling_coefficients_dict_of_dicts )
		)
		unscaling_matrix = 🐢unscaling_matrix
	end

	@set_if_needed( scaling_matrix, kw_arg_symbols,
		_matrix_from_coeff_dict( var_inds, unscaling_coefficients_dict_of_dicts )
	)	

	@set_if_needed( unscaling_matrix, kw_arg_symbols,
		_matrix_from_coeff_dict( var_inds, unscaling_coefficients_dict_of_dicts )
	)

	@set_if_needed( scaling_coefficients_dict_of_dicts, kw_arg_symbols,
		dictionary( vi => _matrix_to_coeff_dict( scaling_matrix, variables, vi_pos ) for (vi_pos,vi) = enumerate(variables) )
	)
	
	@set_if_needed( unscaling_coefficients_dict_of_dicts, kw_arg_symbols,
		dictionary( vi => _matrix_to_coeff_dict( unscaling_matrix, variables, vi_pos ) for (vi_pos,vi) = enumerate(variables) )
	)
	
	n_vars = length(variable_indices)
	for _arg in [ 
		🐢scaling_coefficients_dict_of_dicts, 
		🐢unscaling_coefficients_dict_of_dicts,
		🐢scaling_constants_dict,	
		🐢scaling_constants_vector,
		]
		@check_turtle_length _arg n_vars
	end
	for _arg in [ 
		🐢scaling_matrix,
		🐢unscaling_matrix
		]
		@check_turtle_mat _arg n_vars
	end
	return __init( scal_type;
		variables = 🐢variables,
		variable_indices = var_inds,
		scaling_coefficients_dict_of_dicts = 🐢scaling_coefficients_dict_of_dicts,
		scaling_matrix = 🐢scaling_matrix, 
		scaling_constants_dict = 🐢scaling_constants_dict,
		scaling_constants_vector = 🐢scaling_constants_vector, 
		unscaling_coefficients_dict_of_dicts = 🐢unscaling_coefficients_dict_of_dicts, 
		unscaling_matrix = 🐢unscaling_matrix
	)
end

# TODO: partial jacobian, gradients ?

# ## Implementations
struct SimpleScaler{
	T<:Number,V,N
} <: AbstractAffineScaler

	variables :: V

	D :: SMatrix{N,N,T}
	b :: SVector{N,T}

	Dinv :: SMatrix{N,N,T}
end


function __init( :: Type{<:SimpleScaler};
	variables,
	scaling_matrix :: AbstractMatrix{SM},
	scaling_constants_vector :: AbstractVector{B},
	unscaling_matrix :: AbstractMatrix{USM},
	kwargs...
) where{SM,B,USM}
	T = Base.promote_type(SM,B,USM)
	n_vars = size( scaling_matrix, 1)
	VT = SVector{n_vars,T}
	MT = SMatrix{n_vars,n_vars,T}
	return SimpleScaler(
		variables,
		convert(MT, scaling_matrix),
		convert(VT, scaling_constants_vector),
		convert(MT, unscaling_matrix),
	)
end

__variables( scal :: SimpleScaler ) = scal.variables

__scaling_matrix( scal :: SimpleScaler ) = scal.D
__unscaling_matrix( scal :: SimpleScaler ) = scal.Dinv
__scaling_constants_vector( scal :: SimpleScaler ) = scal.b

# TODO: FullScaler that implements all `__` methods
struct FullScaler{
	T <: Number,
	V, N,
} <: AbstractAffineScaler
	variable_indices :: V
	
	scaling_matrix :: SMatrix{N,N,T}
	scaling_constants_vector :: SVector{N,T}
	
	unscaling_matrix :: SMatrix{N,N,T}
	
	scaling_coefficients_dict_of_dicts :: Dictionary{
		VariableIndex, Dictionary{VariableIndex,T}
	}
	unscaling_coefficients_dict_of_dicts :: Dictionary{
		VariableIndex, Dictionary{VariableIndex,T}
	}
	scaling_constants_dict :: Dictionary{VariableIndex, T}
end

function __init( :: Type{<:FullScaler};
	variable_indices, 
	scaling_coefficients_dict_of_dicts,
	scaling_matrix, 
	scaling_constants_dict,
	scaling_constants_vector,
	unscaling_coefficients_dict_of_dicts,
	unscaling_matrix,
	kwargs...
)
	T = Base.promote_eltype(
		scaling_matrix, 
		unscaling_matrix,
		scaling_constants_vector,
		scaling_constants_dict,
		eltype(scaling_coefficients_dict_of_dicts),
		eltype(unscaling_coefficients_dict_of_dicts),
	)

	n_vars = size( scaling_matrix, 1)
	DT = Dictionary{VariableIndex, Dictionary{VariableIndex, T}}
	MT = SMatrix{n_vars, n_vars, T}
	VT = SVector{n_vars, T}
	return FullScaler(
		variable_indices,
		convert(MT, scaling_matrix), 
		convert(VT, scaling_constants_vector),
		convert(MT, unscaling_matrix),
		convert(DT, scaling_coefficients_dict_of_dicts),
		convert(DT, unscaling_coefficients_dict_of_dicts),
		convert(Dictionary{VariableIndex, T}, scaling_constants_dict),
	)
end

struct FullScalerCustomTypes{
	V,
	T <: Real,
	VT <: AbstractVector{T},
	MT <: AbstractMatrix{T},
	MT2 <: AbstractMatrix{T},
} <: AbstractAffineScaler
	
	variable_indices :: V	
	scaling_matrix :: MT
	scaling_constants_vector :: VT
	unscaling_matrix :: MT2
	scaling_coefficients_dict_of_dicts :: Dictionary{
		VariableIndex, Dictionary{VariableIndex,T}
	}
	unscaling_coefficients_dict_of_dicts :: Dictionary{
		VariableIndex, Dictionary{VariableIndex,T}
	}
	scaling_constants_dict :: Dictionary{VariableIndex, T}
end

function FullScalerCustomTypes(
	variable_indices,
	scaling_matrix, 
	scaling_constants_vector, 
	unscaling_matrix,
	scaling_coefficients_dict_of_dicts,
	unscaling_coefficients_dict_of_dicts,
	scaling_constants_dict,
	kwargs...
)
	T = Base.promote_eltype(
		scaling_matrix, 
		unscaling_matrix,
		scaling_constants_vector,
		scaling_constants_dict,
		eltype(scaling_coefficients_dict_of_dicts),
		eltype(unscaling_coefficients_dict_of_dicts),
	)

	DT = Dictionary{VariableIndex, Dictionary{VariableIndex, T}}
	
	MT = _mat_type( scaling_matrix, T)
	MT2 = _mat_type( unscaling_matrix, T)
	VT = _vec_type(
		scaling_constants_vector, 
		T
	)
	return FullScalerCustomTypes(
		variable_indices,
		convert( MT, scaling_matrix), 
		convert( VT, scaling_constants_vector),
		convert( MT2, unscaling_matrix),
		convert( DT, scaling_coefficients_dict_of_dicts),
		convert( DT, unscaling_coefficients_dict_of_dicts),
		convert( Dictionary{VariableIndex, T}, scaling_constants_dict),
	)
end

function __init( :: Type{<:FullScalerCustomTypes};
	variable_indices, 
	scaling_coefficients_dict_of_dicts,
	scaling_matrix,
	scaling_constants_dict,
	scaling_constants_vector,
	unscaling_coefficients_dict_of_dicts,
	unscaling_matrix,
	kwargs...
)
	return FullScalerCustomTypes(
		variable_indices,
		scaling_matrix, 
		scaling_constants_vector, 
		unscaling_matrix,
		scaling_coefficients_dict_of_dicts,
		unscaling_coefficients_dict_of_dicts,
		scaling_constants_dict,
		)
end

const FSCALER = Union{FullScaler, FullScalerCustomTypes}
__variable_indices( scal :: FSCALER ) = scal.variable_indices
__scaling_matrix( scal :: FSCALER ) = scal.scaling_matrix
__unscaling_matrix( scal :: FSCALER ) = scal.unscaling_matrix
__scaling_constants_vector( scal :: FSCALER ) = scal.scaling_constants_vector
__scaling_coefficients( scal::FSCALER, vi :: ScalarIndex ) = getindex(scal.scaling_coefficients_dict_of_dicts, vi)
__unscaling_coefficients( scal::FSCALER, vi :: ScalarIndex ) = getindex(scal.unscaling_coefficients_dict_of_dicts, vi)
__scaling_constant( scal::FSCALER, vi :: ScalarIndex ) = getindex(scal.scaling_constants_dict, vi)

#####

struct NoVarScaling{ V } <: AbstractAffineScaler 
	variables :: V
	n_vars :: Int 
end

__init( :: Type{<:NoVarScaling};
	variables,
	kwargs...
) = NoVarScaling(variables, length(variables))

__scaling_matrix(scal :: NoVarScaling) = LinearAlgebra.I( scal.n_vars )
__unscaling_matrix(scal :: NoVarScaling) = _scaling_matrix(scal)
__scaling_constants_vector(scal :: NoVarScaling) = zeros(Bool, scal.n_vars)	# TODO `Bool` sensible here?

# overwrite defaults
transform( x, :: NoVarScaling ) = copy(x) 
untransform( _x, :: NoVarScaling ) = copy(_x)
 
# derived functions and helpers, used by the algorithm:

# from two scalers 
# s(x) = Sx + a ⇔ S⁻¹( s(x) - a )
# t(x) = Tx + b
# Return the linear scaler t ∘ s⁻¹, i.e., the 
# scaler that untransforms via s and then applies t:
# TS⁻¹( s(x) - a ) + b
function compose_with_inverse_scaler(
	scal1 :: AbstractAffineScaler, scal2 :: AbstractAffineScaler, 
	target_type = SimpleScaler
)

	var_inds = _variable_indices( scal1 )
	vars_inds2 = _variable_indices( scal2 )
	@assert var_inds == vars_inds2 || isempty(setdiff( var_inds, vars_inds2 ))

	# indices such that variables == vars2[ind2]
	ind2 = if var_inds != vars_inds2
		[ findfirst(v, var_inds) for v = var_inds2 ]
	else
		1:length(var_inds)
	end
	
	scal_mat2 = _scaling_matrix(scal2)[ind2, ind2]
	inv_scal_mat1 =  _unscaling_matrix(scal1)
	scaling_matrix = scal_mat2 * inv_scal_mat1
	scaling_constants_vector = _scaling_constants_vector(scal2)[ind2] - scaling_matrix * _scaling_constants_vector(scal1)

	return _init( target_type;
		variable_indices,
		scaling_matrix,
		scaling_constants_vector,
	)
end

function compose_with_inverse_scaler( scal1 :: NoVarScaling, :: NoVarScaling )
	return scal1 
end
