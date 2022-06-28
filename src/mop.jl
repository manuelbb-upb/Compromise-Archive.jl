# Definitions required:
# * `AbstractMOP`
# * `VariableIndex`
# * `ObjectiveIndex`, `NLConstraintIndexEq`, …
# * `AbstractOuterEvaluator` (evaluators.jl)

Broadcast.broadcastable( mop :: AbstractMOP ) = Ref( mop );

# MANDATORY methods 
# one of:
"Return a vector of `VariableIndice`s used in the model."
__variables(:: AbstractMOP) :: Union{Nothing,AbstractVector{VariableIndex}} = nothing
__variable_indices(::AbstractMOP) :: Union{Nothing,Indices{VariableIndex}} = nothing

"Return the lower bound for the variable with Index `VariableIndex`."
_lower_bound( :: AbstractMOP, :: VariableIndex) :: Real = nothing 
"Return the upper bound for the variable with Index `VariableIndex`."
_upper_bound( :: AbstractMOP, :: VariableIndex) :: Real = nothing

# implement for all supported index types:
"""
    _all_indices( mop, index_type::Type{<:FunctionIndex} )

Return an iterable object of all indices of type `index_type` stored in `mop`.
"""
_all_indices( :: AbstractMOP, T :: Type{<:FunctionIndex} ) = T[]

function _all_indices(mop :: AbstractMOP, T :: Type{FunctionIndex})
    return Iterators.flatten( _all_indices(mop, F) for F in subtypes(T) )
end

# for objectives and nl_constraints (and linear constraints if the "getter"s are defined)
_get( :: AbstractMOP, :: FunctionIndex ) :: AbstractOuterEvaluator = nothing

# Methods for editable models:
# Implement for `mop::AbstractMOP{true}`

function add_variable!(::AbstractMOP)
    @warn("The model does not support adding variables.")
    return nothing
end

for fn in [
    :add_lower_bound!, 
    :add_upper_bound!,
    :del_lower_bound!,
    :del_upper_bound!
]
    @eval function $(fn)(::AbstractMOP, vi :: VariableIndex, bound :: Real )
        @warn("The model does not support adding bounds.")
        return nothing
    end
end

function _add_function!(
    mop :: AbstractMOP, 
    index :: FunctionIndex, 
    func :: Union{AbstractOuterEvaluator}
) :: Bool
    false
end

#=
# The functions that are to be called by the user are derived below.
# They do not use an underscore.
"Add an objective function to the model."
_add_objective!(::AbstractMOP{true}, ::ObjectiveIndex, ::AbstractOuterEvaluator) = nothing

"Add a nonlinear equality constraint function to the model."
_add_nl_eq_constraint!(::AbstractMOP{true}, ::NLConstraintIndexEq, ::AbstractOuterEvaluator) = nothing

"Add a nonlinear inequality constraint function to the model."
_add_nl_ineq_constraint!(::AbstractMOP{true}, ::NLConstraintIndexIneq, ::AbstractOuterEvaluator) = nothing

"Add a linear equality constraint function to the model."
_add_eq_constraint!(::AbstractMOP{true}, ::ConstraintIndexEq, ::MOI.VectorAffineFunction) = nothing
"Add a linear inequality constraint function to the model."
_add_ineq_constraint!(::AbstractMOP{true}, ::ConstraintIndexIneq, :: MOI.VectorAffineFunction) = nothing
=#

function new_index(mop, index_type, name) end
_next_int( value_indices ) = 1 + isempty(value_indices) ? 0 : maximum( ind.value for ind in value_indices )
for index_type in subtypes( FunctionIndex )
    @eval function _new_index( mop :: AbstractMOP, ind_type :: Type{$(index_type)}, name::String="")
        i = _next_int( _all_indices(mop, ind_type) )
        return ind_type( i, name )
    end
end

# not used anywhere yet
# not implemented yet by `MOP<:AbstractMOP{true}`
"Remove a function from the MOP."
function _del!(::AbstractMOP, ind :: Union{VariableIndex, FunctionIndex} )
    @warn "The model does not support deleting the object with $(ind)."
    return nothing
end

# DERIVED methods

#=
"Return a vector or tuple of all objective indices used in the model."
_objective_indices( mop :: AbstractMOP ) = _all_indices( mop, ObjectiveIndex )

# nonlinear constraints
"Return the vector or tuple of the indices of nonlinear equality constraints used in the model."
_nl_eq_constraint_indices( mop :: AbstractMOP ) = _all_indices( mop, NLConstraintIndexEq )
"Return the vector or tuple of the indices of nonlinear inequality constraints used in the model."
_nl_ineq_constraint_indices( mop :: AbstractMOP ) = _all_indices( mop, NLConstraintIndexIneq )

# linear constraints (optional - either define these or 
# `get_eq_matrix_and_vector` as well as `get_ineq_matrix_and_vector`)
"Return the vector or tuple of the indices of *linear* equality constraints used in the model."
_eq_constraint_indices( mop :: AbstractMOP ) = _all_indices( mop, ConstraintIndexEq )
"Return the vector or tuple of the indices of *linear* inequality constraints used in the model."
_ineq_constraint_indices( mop :: AbstractMOP ) = _all_indices( mop, ConstraintIndexIneq )
=#

function _variables(mop :: AbstractMOP)
    vars = __variables(mop)
    isnothing(vars) && return collect( __variable_indices(mop) )
    return vars
end
function _variable_indices(mop :: AbstractMOP)
    vars = __variable_indices(mop)
    isnothing(vars) && return Indices( __variables(mop) )
    return vars
end
_all_indices( mop :: AbstractMOP, :: Type{VariableIndex} ) = _variables(mop)

function _lower_bounds( mop :: AbstractMOP, var_inds )
    return _lower_bound.(mop, var_inds)
end

function _upper_bounds( mop :: AbstractMOP, var_inds )
    return _upper_bound.(mop, var_inds)
end

"Return full vector of lower variable vectors for original problem."
function _lower_bounds_vector( mop :: AbstractMOP )
    return _lower_bounds( mop, _variable_indices(mop) )
end

"Return full vector of upper variable vectors for original problem."
function _lower_bounds( mop :: AbstractMOP )
    return get_upper_bounds( mop, _variable_indices(mop) )
end

# can be improved
num_vars( mop :: AbstractMOP ) :: Int = length(_variable_indices(mop))

function add_variables!(mop::AbstractMOP{true}, num_new :: Int )
    return [add_variable!(mop) for _ = 1 : num_new]
end

function _bounds_vectors( mop :: AbstractMOP )
    (_lower_bounds_vector(mop), full_upper_bounds(mop))
end

function _all_functions( mop :: AbstractMOP, index_type :: Type{<:FunctionIndex} )
    return ( _get( mop, func_ind ) for func_ind = _all_indices(mop, index_type) )
end

function _all_functions( mop :: AbstractMOP )
    return __all_functions(mop, FunctionIndex) 
end

#=
function list_of_objectives( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = _objective_indices(mop) ]
end

function list_of_nl_eq_constraints( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = _nl_eq_constraint_indices(mop) ]
end

function list_of_nl_ineq_constraints( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = _nl_ineq_constraint_indices(mop) ]
end

function list_of_functions( mop :: AbstractMOP )
    return [ _get( mop, func_ind ) for func_ind = _function_indices(mop) ]
end
=#

# defined below:
num_objectives(mop) = nothing
num_eq_constraints(mop) = nothing
num_ineq_constraints(mop) = nothing
num_nl_eq_constraints(mop) = nothing
num_nl_ineq_constraints(mop) = nothing

for func_type in [
    :objective,
    :eq_constraint,
    :ineq_constraint,
    :nl_eq_constraint,
    :nl_ineq_constraint,
]
    fn = Symbol("num_$(func_type)s")
    ind_fn = Symbol("_$(func_type)_indices")
    @eval function $(fn)(mop :: AbstractMOP )
        return sum(
            num_outputs( _get(mop, ind) for ind in $(ind_fn)(mop) )
        )
    end
end

function num_nl_constraints( mop :: AbstractMOP )
    num_nl_eq_constraints(mop) + num_nl_ineq_constraints(mop)
end

function num_lin_constraints( mop :: AbstractMOP )
    num_eq_constraints(mop) + num_ineq_constraints(mop)
end

####### helpers for linear constraints
#=
function add_eq_constraint!( mop :: AbstractMOP{true}, aff_func :: MOI.ScalarAffineFunction )
    return add_eq_econstraint!(mop, _scalar_to_vector_aff_func(aff_func) )
end

function add_ineq_constraint!( mop :: AbstractMOP{true}, aff_func :: MOI.ScalarAffineFunction )
    return add_ineq_econstraint!(mop, _scalar_to_vector_aff_func(aff_func) )
end

"""
    add_ineq_constraint!(mop, A, b = [])

Add linear inequality constraints ``Ax ≤ b`` to the problem `mop`.
If `b` is empty, then ``Ax ≤ 0`` is added.
Returns a `ConstraintIndexIneq`.
"""
function add_ineq_constraint!(mop :: AbstractMOP{true}, A :: AbstractMatrix, b :: AbstractVector = [], vars :: Union{Nothing,AbstractVector{<:VariableIndex}} = nothing)
	_vars = isnothing( vars ) ? _variable_indices(mop) : vars
    _b = isempty( b ) ? zeros( Bool, size(A,1) ) : -b
	return _add_ineq_constraint!(mop, 
		_matrix_to_vector_affine_function( A, _b; variables = _vars )
	)
end

"""
    add_eq_constraint!(mop, A, b = [])

Add linear equality constraints ``Ax = b`` to the problem `mop`.
If `b` is empty, then ``Ax = 0`` is added.
Returns a `ConstraintIndexEq`.
"""
function add_eq_constraint!(mop :: AbstractMOP{true}, A :: AbstractMatrix, b :: AbstractVector, vars :: Union{Nothing,AbstractVector{<:VariableIndex}} = nothing)
	_vars = isnothing( vars ) ? _variable_indices(mop) : vars
    _b = isempty( b ) ? zeros( Bool, size(A,1) ) : -b
	return _add_ineq_constraint!(mop, 
		_matrix_to_vector_affine_function( A, _b; variables = _vars )
	)
end
=#

# pretty printing
_is_editable(::Type{<:AbstractMOP{T}}) where T = T 

function Base.show(io::IO, mop :: M) where M<:AbstractMOP
    str = "$( _is_editable(M) ? "Editable" : "Non-editable") MOP of Type $(_typename(M)). "
    if !get(io, :compact, false)
        str *= """There are 
        * $(num_vars(mop)) variables and $(num_objectives(mop)) objectives,
        * $(num_nl_eq_constraints(mop)) nonlinear equality and $(num_nl_ineq_constraints(mop)) nonlinear inequality constraints,
        * $(num_eq_constraints(mop)) linear equality and $(num_ineq_constraints(mop)) linear inequality constraints.
        The lower bounds and upper variable bounds are 
        $(_prettify(_lower_bounds_vector(mop), 5))
        $(_prettify(full_upper_bounds(mop), 5))"""
    end
    print(io, str)
end

#### utilities

function check_variable_bounds(mop :: AbstractMOP, xd :: AbstractDictionary{ScalarIndex,<:Real} )
    var_inds = _variable_indices(mop)
    for (vi,xi) in pairs(getindices(xd,var_inds))
        if _lower_bound(mop, vi) > xi || _upper_bound(mop, vi) < xi 
            return false
        end
    end
    return true 
end

function project_into_bounds!(xd :: AbstractDictionary{ScalarIndex,<:Real}, mop :: AbstractMOP ) 
    var_inds = _variable_indices(mop)
    for (vi, xi) in pairs(getindices(xd, var_inds))
        set!(x, vi, 
            min( 
                _upper_bound( mop, vi), 
                max( 
                    _lower_bound(mop, vi),
                    xi
                ) 
            )
        )
    end
    return nothing
end

# Evaluation

function eval_at_unscaled_dict(
    mop :: AbstractMOP, xd :: AbstractDictionary{<:ScalarIndex,F}
) where F
    _xd = convert( Dictionary{ScalarIndex, F}, xd )
    for func_ind in _all_indices(mop, FunctionIndex)
        eval_at_dict!(_get(mop, func_ind), _xd)
    end
    return _xd
end

#=======================================================================
SimpleMOP
=======================================================================#
const MOP_FIELDNAME_DICT_SG = Dict(
    VariableIndex => :variable,
    ObjectiveIndex => :objective,
    ConstraintIndexEq => :eq_constraint,
    ConstraintIndexIneq => :ineq_constraint,
    NLConstraintIndexEq => :nl_eq_constraint,
    NLConstraintIndexIneq => :nl_ineq_constraint,
)
_plural( symb ) = Symbol( symb, :s )
const MOP_FIELDNAME_DICT = Dict( k => _plural(v) for (k,v) in pairs(MOP_FIELDNAME_DICT_SG) )

@with_kw struct SimpleMOP <: AbstractMOP{true}
    variables :: Vector{VariableIndex}
    
    lower_bounds_dict = FillDictionary( variables, -Inf )
    upper_bounds_dict = FillDictionary( variables, Inf )

    objectives = Dictionary{ObjectiveIndex,AbstractOuterEvaluator}()
    nl_eq_constraints = Dictionary{NLConstraintIndexEq,AbstractOuterEvaluator}()
    nl_ineq_constraints = Dictionary{NLConstraintIndexIneq,AbstractOuterEvaluator}()
    eq_constraints = Dictionary{ConstraintIndexEq,AbstractOuterEvaluator}()
    ineq_constraints = Dictionary{ConstraintIndexIneq,AbstractOuterEvaluator}()
end

for (ind_type, field_name) in pairs( MOP_FIELDNAME_DICT )
    @eval function _add_function!(
            mop :: SimpleMOP, ind :: $(ind_type), aoe :: AbstractOuterEvaluator
        )
        insert!( getfield( mop, $(Meta.quot(field_name))), ind, aoe )
        return nothing
    end
end

# syntactic sugar:

function _get_differentiator(model_cfg, gradients, jacobian, hessians, ad_backend, ad_backend_2 )
    if !needs_gradients(model_cfg) && !needs_hessians(model_cfg)
        return nothing
    end
    if needs_gradients(model_cfg)
        if isnothing(gradients) && isnothing(jacobian) && isnothing(ad_backend)
            error("`model_cfg` requires gradients but `gradients`, `jacobian` and `ad_backend` are nothing.")
        end
    end
    if needs_hessians(model_cfg)
        if isnothing(hessians) && isnothing(ad_backend) && isnothing(ad_backend2)
            error("`model_cfg` requires hessians but `hessians`, `ad_backend` and `ad_backend2` are nothing.")
        end
    end

    return FuncContainerBackend(;
        gradients, jacobian, hessians, fallback = ad_backend, fallback2 = ad_backend2
    )
end
#=
for (func_type, ind_type) in [
    (:objective, ObjectiveIndex),
    (:nl_eq_constraint, NLConstraintIndexEq),
    (:nl_ineq_constraint, NLConstraintIndexIneq)
]
    fn = Symbol("add_$(func_type)!")
    ind_fn = Symbol("_$(func_type)_indices")
    @eval function $(fn)(mop :: AbstractMOP{true}, func :: Function;
        n_out :: Int, 
        model_cfg :: AbstractSurrogateConfig = DUMMY_CONFIG,
        can_batch :: Bool = false,
        gradients = nothing,
        jacobian = nothing, 
        hessians = nothing,
        ad_backend = nothing,
        ad_backend_second_order = nothing,
    )
        differentiator = _get_differentiator(
            model_cfg, gradients, jacobian, hessians, ad_backend, ad_backend_second_order
        )
        aie = WrappedUserFunc(func;
            num_outputs = n_out,
            model_cfg, can_batch, differentiator
        )

        insert!( getfield( mop, $(Meta.quot(fieldn))), ind, aoe )
        return nothing
    end
end
=#