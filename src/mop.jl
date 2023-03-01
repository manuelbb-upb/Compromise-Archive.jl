# Definitions required:
# * `AbstractMOP`
# * `VariableIndex`
# * `ObjectiveIndex`, `NLConstraintIndexEq`, …
# * `AbstractOuterEvaluator` (evaluators.jl)

Broadcast.broadcastable(mop::AbstractMOP) = Ref(mop);

# MANDATORY methods 
# one of:
"Return a vector of `VariableIndice`s used in the model."
__variables(::AbstractMOP)::Union{Nothing,AbstractVector{VariableIndex}} = nothing
__variable_indices(::AbstractMOP)::Union{Nothing,Indices{VariableIndex}} = nothing

"Return the lower bound for the variable with Index `VariableIndex`."
_lower_bound(::AbstractMOP, ::VariableIndex)::Real = nothing
"Return the upper bound for the variable with Index `VariableIndex`."
_upper_bound(::AbstractMOP, ::VariableIndex)::Real = nothing

# optional:
_primal_value(::AbstractMOP, ::VariableIndex)::Real = MIN_PRECISION(NaN)
function _primal_value(::AbstractMOP, ::FunctionIndex)::Union{Real,AbstractVector{<:Real}}
    return MIN_PRECISION(NaN)
end

# implement for all supported index types:
"""
    _all_indices( mop, index_type::Type{<:FunctionIndex} )

Return an iterable object of all indices of type `index_type` stored in `mop`.
"""
_all_indices(::AbstractMOP, T::Type{<:FunctionIndex}) = T[]

function _all_indices(mop::AbstractMOP, T::Type{FunctionIndex})
    return Iterators.flatten(_all_indices(mop, F) for F in subtypes(T))
end

function _get(::AbstractMOP, ::FunctionIndex)::Union{Nothing,AbstractOuterEvaluator}
    return nothing
end

# Methods for editable models:
# Implement for `mop::AbstractMOP{true}`

function add_variable!(::AbstractMOP, args...)
    @warn("The model does not support adding variables.")
    return nothing
end

for fn in [
    :add_lower_bound!,
    :add_upper_bound!,
    :set_primal!
]
    @eval function $(fn)(::AbstractMOP, vi::VariableIndex, bound::Real)
        @warn("The model does not support setting primals or adding bounds.")
        return nothing
    end
end
for fn in [
    :del_lower_bound!,
    :del_upper_bound!,
    :del_primal!
]
    @eval function $(fn)(::AbstractMOP, vi::VariableIndex)
        @warn("The model does not support un-setting primals or adding bounds.")
        return nothing
    end
end

function _add_function!(
    mop::AbstractMOP,
    index::FunctionIndex,
    func::Union{AbstractOuterEvaluator}
)::Bool
    false
end

function set_primal_at_index!(::AbstractMOP, ::DependentIndex, ::Real)
    @warn("The model dos not support setting primal values for stored functions.")
    return nothing
end
function del_primal_at_index!(::AbstractMOP, ::DependentIndex)
    @warn("The model dos not support setting primal values for stored functions.")
    return nothing
end

# (helper)
function _get_func_and_inds_or_error(mop, ind)
    if !_contains(mop, ind)
        error("The problem `mop` has no function with index $(ind).")
    end
    func = _get(mop, ind)
    dep_inds = output_indices(func)
    return func, dep_inds
end

function set_primal!(mop::AbstractMOP, ind::FunctionIndex, value::Real)
    _, dep_inds = _get_func_and_inds_or_error(mop, ind)
    if length(dep_inds) > 1
        error("Trying to set a single value for a multi-dimensional function. Use `set_primals!` instead.")
    end
    set_primal_at_index!(mop, first(dep_inds), value)
    return nothing
end

function set_primals!(
    mop::AbstractMOP,
    ind::FunctionIndex,
    values::AbstractVector{<:Real}
)
    _, dep_inds = _get_func_and_inds_or_error(mop, ind)
    if length(dep_inds) != length(values)
        error("Number of supplied values does not match output dimension of $(ind).")
    end
    set_primal_at_index!.(mop, ind, values)
    return nothing
end

function set_primals!(
    mop::AbstractMOP,
    ind::FunctionIndex,
    val_dict::AbstractDictionary{DependentIndex,<:Real}
)
    _, dep_inds = _get_func_and_inds_or_error(mop, ind)
    if !issubset(dep_inds, keys(val_dict))
        error("Not all output_indices for $(ind) present in keys of `val_dict`.")
    end
    for di in dep_inds
        set_primal_at_index!.(mop, di, getindex(val_dict, di))
    end
    return nothing
end

function del_primals!(mop::AbstractMOP, ind::FunctionIndex)
    _, dep_inds = _get_func_and_inds_or_error(mop, ind)
    for di in dep_inds
        del_primal_at_index!(mop, di)
    end
    return nothing
end
function del_primal!(mop::AbstractMOP, ind::FunctionIndex)
    return del_primals!(mop, ind)
end

function del_primal!(mop::AbstractMOP, ind::DependentIndex)
    return del_primal_at_index!(mop, ind)
end
function del_primals!(mop::AbstractMOP, ind::DependentIndex)
    return del_primal_at_index!(mop, ind)
end

# Helper function `_new_index(mop, ind_type, name="")`
# to get a new index of type `ind_type` for `mop`.
function _new_index(mop, index_type, name) end

function _next_int(value_indices)
    isempty(value_indices) && return 1
    return 1 + maximum(ind.value for ind in value_indices)
end

for index_type in [subtypes(FunctionIndex); VariableIndex]
    @eval function _new_index(mop::AbstractMOP, ind_type::Type{$(index_type)}, name::String="")
        i = _next_int(_all_indices(mop, ind_type))
        return ind_type(i, name)
    end
end

# not used anywhere yet
# not implemented yet by `MOP<:AbstractMOP{true}`
"Remove a function from the MOP."
function _del!(::AbstractMOP, ind::Union{VariableIndex,FunctionIndex})
    @warn "The model does not support deleting the object with $(ind)."
    return nothing
end

# DERIVED methods

function _variables(mop::AbstractMOP)
    vars = __variables(mop)
    isnothing(vars) && return collect(__variable_indices(mop))
    return vars
end
function _variable_indices(mop::AbstractMOP)
    vars = __variable_indices(mop)
    isnothing(vars) && return Indices(__variables(mop))
    return vars
end
_all_indices(mop::AbstractMOP, ::Type{VariableIndex}) = _variables(mop)

function _has_primal_value(mop::AbstractMOP, vi::VariableIndex)
    return !isnan(_primal_value(mop, vi))
end
function _has_primal_value(mop::AbstractMOP, fi::FunctionIndex)
    return !isnan(_primal_value(mop, fi))
end
function _has_primal_variable_values(mop::AbstractMOP)
    return all(_has_primal_value(mop, vi) for vi in _variables(mop))
end
function _has_primal_evaluation_values(mop::AbstractMOP)
    return all(_has_primal_value(mop, fi) for fi in _all_indices(mop, FunctionIndex))
end
function _primal_variable_vector(mop::AbstractMOP)
    return collect(_primal_value(mop, vi) for vi in _variables(mop))
end
function _primal_evaluation_vector(mop::AbstractMOP)
    return collect(_primal_value(mop, fi) for fi in _all_indices(mop, FunctionIndex))
end

function _primal_values(mop::AbstractMOP, ind::FunctionIndex)
    func, dep_inds = _get_func_and_inds_or_error(mop, ind)
    return dictionary(di => _primal_value(mop,di) for di in dep_inds )
end

# might be improved by implementations
function _contains(mop::AbstractMOP, ind::FunctionIndex)
    return !isnothing(_get(mop, ind))
end

# might be improved by implementations
function _contains(mop::AbstractMOP, ind::VariableIndex)
    return haskey(_variable_indices(mop), ind)
end

function _lower_bounds(mop::AbstractMOP, var_inds)
    return _lower_bound.(mop, var_inds)
end

function _upper_bounds(mop::AbstractMOP, var_inds)
    return _upper_bound.(mop, var_inds)
end

"Return full vector of lower variable vectors for original problem."
function _lower_bounds_vector(mop::AbstractMOP)
    return _lower_bounds(mop, _variable_indices(mop))
end

"Return full vector of upper variable vectors for original problem."
function _upper_bounds_vector(mop::AbstractMOP)
    return _upper_bounds(mop, _variable_indices(mop))
end

# can be improved
num_vars(mop::AbstractMOP)::Int = length(_variable_indices(mop))

function add_variables!(mop::AbstractMOP{true}, num_new::Int)
    return [add_variable!(mop) for _ = 1:num_new]
end

function _bounds_vectors(mop::AbstractMOP)
    (_lower_bounds_vector(mop), _upper_bounds_vector(mop))
end

function _all_functions(mop::AbstractMOP, index_type::Type{<:FunctionIndex})
    return (_get(mop, func_ind) for func_ind = _all_indices(mop, index_type))
end

function _all_functions(mop::AbstractMOP)
    return _all_functions(mop, FunctionIndex)
end

# pretty printing
_is_editable(::Type{<:AbstractMOP{T}}) where {T} = T
_is_editable(::T) where {T<:AbstractMOP} = _is_editable(T)

function Base.show(io::IO, mop::T) where {T<:AbstractMOP}
    str = if _is_editable(mop)
        "Editable"
    else
        "Non-editable"
    end
    str *= " MOP of type `$(Base.typename(T).name)`."
    if !get(io, :compact, false)
        str *= """ There are 
       * $(num_vars(mop)) variables and $(dim_objectives(mop)) objective outputs,
       * $(dim_nl_eq_constraints(mop)) nonlinear equality and $(dim_nl_ineq_constraints(mop)) nonlinear inequality constraints,
       * $(dim_eq_constraints(mop)) linear equality and $(dim_ineq_constraints(mop)) linear inequality constraints.
       The lower bounds and upper variable bounds are 
       $(_prettify(_lower_bounds_vector(mop); digits = 4))
       $(_prettify(_upper_bounds_vector(mop); digits = 4))
        """
    end
    print(io, str)
end

#### utilities

function check_variable_bounds(mop::AbstractMOP, xd::AbstractDictionary{<:ScalarIndex,<:Real})
    var_inds = _variable_indices(mop)
    for (vi, xi) in pairs(getindices(xd, var_inds))
        if _lower_bound(mop, vi) > xi || _upper_bound(mop, vi) < xi
            return false
        end
    end
    return true
end

function project_into_bounds!(xd::AbstractDictionary{<:ScalarIndex,<:Real}, mop::AbstractMOP)
    var_inds = _variable_indices(mop)
    for (vi, xi) in pairs(getindices(xd, var_inds))
        set!(xd, vi,
            min(
                _upper_bound(mop, vi),
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

function eval_at_dict!(
    mop::AbstractMOP, xd::AbstractDictionary{>:SCALAR_INDEX,F}
) where {F}

    for func_ind in _all_indices(mop, FunctionIndex)
        eval_at_dict!(_get(mop, func_ind), xd)
    end
    return xd
end

function extract_vector(
    x0::AbstractDictionary{<:ScalarIndex,F},
    indices,
) where {F}
    return collect(F, getindices(x0, indices))
end

function extract_vector(
    mop::AbstractMOP,
    x0::AbstractDictionary,
    ind_type::Type{<:VariableIndex}
)
    return extract_vector(x0, _all_indices(mop, ind_type))
end

function extract_vector(
    mop::AbstractMOP,
    x0::AbstractDictionary,
    ind_type::Type{<:FunctionIndex}
)
    return reduce(
        vcat,
        extract_vector(
             x0,
             output_indices(func)
        ) for func in _all_functions(mop, ind_type)
    )
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
_plural(symb) = Symbol(symb, :s)
const MOP_FIELDNAME_DICT = Dict(k => _plural(v) for (k, v) in pairs(MOP_FIELDNAME_DICT_SG))

const MOP_FIELDNAME_NOUN_DICT = Dict(
    VariableIndex => "variable",
    ObjectiveIndex => "objective function",
    ConstraintIndexEq => "linear equality constraint function",
    ConstraintIndexIneq => "linear inequality constraint function",
    NLConstraintIndexEq => "nonlinear equality constraint function",
    NLConstraintIndexIneq => "nonlinear inequality constraint function",
)
_plural(s::String) = s * "s"

# defined in for-loop below:
dim_objectives(mop) = nothing
dim_eq_constraints(mop) = nothing
dim_ineq_constraints(mop) = nothing
dim_nl_eq_constraints(mop) = nothing
dim_nl_ineq_constraints(mop) = nothing
for (ind_type, fn) in pairs(MOP_FIELDNAME_DICT)
    if ind_type != VariableIndex
        fn = Symbol("dim_$(fn)")
        @eval begin
            @doc "Dimension of the concatenated output vector of $($(_plural(MOP_FIELDNAME_NOUN_DICT[ind_type])))."
            function $(fn)(mop::AbstractMOP)
                ind = _all_indices(mop, $ind_type)
                isempty(ind) && return 0
                return sum(
                    num_outputs(_get(mop, ind)) for ind in _all_indices(mop, $ind_type)
                )
            end
        end
    end
end

function dim_nl_constraints(mop::AbstractMOP)
    return dim_nl_eq_constraints(mop) + dim_nl_ineq_constraints(mop)
end

function num_lin_constraints(mop::AbstractMOP)
    return dim_eq_constraints(mop) + dim_ineq_constraints(mop)
end

Base.@kwdef struct SimpleMOP <: AbstractMOP{true}
    variables::Vector{VariableIndex} = []

    primal_values::Dictionary{VariableIndex,Float64} = Dictionary()
    lower_bounds::Dictionary{VariableIndex,Float64} = Dictionary()
    upper_bounds::Dictionary{VariableIndex,Float64} = Dictionary()

    objectives = Dictionary{ObjectiveIndex,AbstractOuterEvaluator}()
    nl_eq_constraints = Dictionary{NLConstraintIndexEq,AbstractOuterEvaluator}()
    nl_ineq_constraints = Dictionary{NLConstraintIndexIneq,AbstractOuterEvaluator}()
    eq_constraints = Dictionary{ConstraintIndexEq,AbstractOuterEvaluator}()
    ineq_constraints = Dictionary{ConstraintIndexIneq,AbstractOuterEvaluator}()

    primal_evals::Dictionary{DependentIndex,Float64} = Dictionary()
end

__variable_indices(mop::SimpleMOP) = keys(mop.lower_bounds)
__variables(mop::SimpleMOP) = mop.variables
_lower_bound(mop::SimpleMOP, vi::VariableIndex) = getindex(mop.lower_bounds, vi)
_lower_bounds(mop::SimpleMOP, var_inds) = collect(getindices(mop.lower_bounds, var_inds))
_upper_bound(mop::SimpleMOP, vi::VariableIndex) = getindex(mop.upper_bounds, vi)
_upper_bounds(mop::SimpleMOP, var_inds) = collect(getindices(mop.upper_bounds, var_inds))
_primal_value(mop::SimpleMOP, vi::VariableIndex) = getindex(mop.primal_values, vi)
_primal_value(mop::SimpleMOP, vi::DependentIndex) = getindex(mop.primal_evals, vi)

function add_lower_bound!(mop::SimpleMOP, vi::VariableIndex, bound::Real)
    set!(mop.lower_bounds, vi, bound)
end
function add_upper_bound!(mop::SimpleMOP, vi::VariableIndex, bound::Real)
    set!(mop.upper_bounds, vi, bound)
end
function del_lower_bound!(mop::SimpleMOP, vi::VariableIndex)
    set!(mop.lower_bounds, vi, -Inf)
end
function del_upper_bound!(mop::SimpleMOP, vi::VariableIndex)
    set!(mop.upper_bounds, vi, Inf)
end

function set_primal!(mop::SimpleMOP, vi::VariableIndex, bound::Real)
    set!(mop.primal_values, vi, bound)
end
function del_primal!(mop::SimpleMOP, vi::VariableIndex)
    set!(mop.primal_values, vi, NaN)
end

function del_primal_at_index!(mop::SimpleMOP, di::DependentIndex)
    delete!(mop.primal_evals, di)
    return nothing
end

function set_primal_at_index!(mop::SimpleMOP, di::DependentIndex, val::Real)
    set!(mop.primal_evals, di, val)
    return nothing
end

function _del!(mop::SimpleMOP, vi::VariableIndex)
    isempty(mop.variables) && return nothing
    i = findfirst(isequal(vi), mop.variables)
    if !isnothing(vi)
        deleteat!(mop.variables, i)
        delete!(mop.lower_bounds, vi)
        delete!(mop.upper_bounds, vi)
        delete!(mop.primal_values, vi)
    end
    return nothing
end

for (ind_type, fn) in pairs(MOP_FIELDNAME_DICT)
    @eval begin
        function _all_indices(mop::SimpleMOP, ind_type::$(ind_type))
            return keys(getfield(mop, $(Meta.quot(fn))))
        end
    end
end

function add_variable!(mop::SimpleMOP;
    lb::Real=-Inf, ub::Real=Inf, name::String=""
)
    new_ind = _new_index(mop, VariableIndex, name)
    push!(mop.variables, new_ind)
    set!(mop.lower_bounds, new_ind, lb)
    set!(mop.upper_bounds, new_ind, ub)
    set!(mop.primal_values, new_ind, NaN)
    return new_ind
end

function add_variable!(mop::SimpleMOP, vi::VariableIndex; lb=-Inf, ub=Inf)
    if !(vi in mop.variables)
        push!(mop.variables, vi)
        set!(mop.lower_bounds, vi, lb)
        set!(mop.upper_bounds, vi, ub)
    end
    return nothing
end

# `add_function(mop, ind, aoe)` and
# `_get(mop, ind)`
# `set_primal!`
# `del_primal!`
# for all function index types
for (ind_type, field_name) in pairs(MOP_FIELDNAME_DICT)
    if ind_type != VariableIndex
        @eval begin
            function _all_indices(mop::SimpleMOP, ::Type{<:$(ind_type)})
                return keys(getfield(mop, $(Meta.quot(field_name))))
            end

            function _add_function!(
                mop::SimpleMOP, ind::$(ind_type), aoe::AbstractOuterEvaluator
            )
                insert!(getfield(mop, $(Meta.quot(field_name))), ind, aoe)
                return nothing
            end
            function _get(mop::SimpleMOP, ind::$(ind_type))
                return getindex(getfield(mop, $(Meta.quot(field_name))), ind)
            end
            function _del!(mop::SimpleMOP, ind::$(ind_type))
                delete!(getfield(mop, $(Meta.quot(field_name))), ind)
                return nothing
            end

        end
    end
end

# syntactic sugar:

function _get_differentiator(model_cfg, gradients, jacobian, hessians, ad_backend, ad_backend2)
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
        gradients, jacobian, hessians, fallback=ad_backend, fallback2=ad_backend2
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
