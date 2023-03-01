# depends on 
# * `indices.jl` (`ScalarIndex`, etc.)

"""
	@forward_pipe T, get_inner, method1, method2, ...

For each method `m` in the list of methods, define a method `m`
that exepts an object `x` of type `T` as its first argument and
passes all other `args...` and `kwargs...` done to
`m( get_inner(x), args...; kwargs...)`.
"""
macro forward_pipe(typename_ex, extractor_method_ex, fs)
    T = esc(typename_ex)
    getter = esc(extractor_method_ex)
    fs = isexpr(fs, :tuple) ? map(esc, fs.args) : [esc(fs)]
    :($([:($f(x::$T, args...; kwargs...) = (Base.@_inline_meta; $f($(getter)(x), args...; kwargs...)))
         for f in fs]...);
    nothing)
end

"""
	@forward_pipes T, get_inner, method1, method2, ...

For each method `m` in the list of methods, define a method `m`
that exepts an object `x` of type `T` as its first argument,
some index `l` as its second argument and
passes all other `args...` and `kwargs...` done to
`m( get_inner(x, l), args...; kwargs...)`.
"""
macro forward_pipes(typename_ex, extractor_method_ex, fs)
    T = esc(typename_ex)
    getter = esc(extractor_method_ex)
    fs = isexpr(fs, :tuple) ? map(esc, fs.args) : [esc(fs)]
    :($([:($f(x::$T, sub_index, args...; kwargs...) = (Base.@_inline_meta; $f($(getter)(x, sub_index), args...; kwargs...)))
         for f in fs]...);
    nothing)
end

Base.convert(T::Type{<:AbstractIndices{I}}, x::AbstractVector{I}) where {I} = T(x)
Base.convert(T::Type{<:AbstractVector{I}}, x::AbstractIndices{I}) where {I} = T(x)

#=====================================================================
AbstractInnerEvaluator
=====================================================================#

# `AbstractInnerEvaluator` interface
# This interface meant to be implemented by surrogate models and
# user provided functions alike. The latter will likely be wrapped in other
# types to make the below methods availabe.
# An `AbstractInnerEvaluator` is a “mathematical” object:
# It takes real-valued vectors as input and returns real-valued vectors for output.
# The mapping of variable dictionaries to vectors happens elsewhere.
Base.broadcastable(ev::AbstractInnerEvaluator) = Ref(ev)

#_num_inputs( :: AbstractInnerEvaluator ) :: Int = 0
# (mandatory)
num_outputs(::AbstractInnerEvaluator)::Int = 0

# ##################### counter helpers #########################

# (optional) used for stopping when user provided functions reach a limit
struct EmptyRefCounter end
const EMPTY_COUNTER = EmptyRefCounter()

num_eval_counter(::AbstractInnerEvaluator)::Union{EmptyRefCounter,Base.RefValue{Int}} = EMPTY_COUNTER

Base.getindex(::EmptyRefCounter) = 0
Base.setindex!(::EmptyRefCounter, x) = nothing

function increase_counter!(c::Union{EmptyRefCounter,Base.RefValue{Int}}, N=1)
    c[] += N
    return nothing
end

function set_counter!(c::Union{EmptyRefCounter,Base.RefValue{Int}}, N=0)
    c[] = N
    return nothing
end

function increase_counter!(ov::AbstractInnerEvaluator, N=1)
    return increase_counter!(num_eval_counter(ov), N)
end

function set_counter!(ov::AbstractInnerEvaluator, N=1)
    return set_counter!(num_eval_counter(ov), N)
end

function num_evals(ov::AbstractInnerEvaluator)
    return num_eval_counter(ov)[]
end
# #################################################################

# (mandatory)
_eval_at_vec(::AbstractInnerEvaluator, ::Vec)::Vec = nothing

# derived defaults:
function _eval_at_vec(ev::AbstractInnerEvaluator, x::Vec, output_number)
    return _eval_at_vec(ev, x)[output_number]
end

function _eval_at_vecs(ev::AbstractInnerEvaluator, X::VecVec)
    return _eval_at_vec.(ev, X)
end

function _eval_at_vecs(ev::AbstractInnerEvaluator, X::VecVec, output_number)
    return [v[output_number] for v in _eval_at_vecs(ev, X, output_number)]
end

function eval_at_vec(ev::AbstractInnerEvaluator, args...)
    increase_counter!(ev)
    return _eval_at_vec(ev, args...)
end

function eval_at_vecs(ev::AbstractInnerEvaluator, X::VecVec, args...)
    increase_counter!(ev, length(X))
    return _eval_at_vecs(ev, X, args...)
end

"Returns the integer `n` for which the `n`-th derivative of `aie` is constant."
_diff_order(::AbstractInnerEvaluator) = Val(-1)

_provides_gradients(aie::AbstractInnerEvaluator)::Bool = false
_provides_jacobian(aie::AbstractInnerEvaluator)::Bool = false
_provides_hessians(aie::AbstractInnerEvaluator)::Bool = false

_gradient(::AbstractInnerEvaluator, x::Vec; output_number) = nothing
_jacobian(::AbstractInnerEvaluator, x::Vec) = nothing
_partial_jacobian(::AbstractInnerEvaluator, ::Vec; output_numbers) = nothing
_hessian(::AbstractInnerEvaluator, ::Vec; output_number) = nothing
_jacobian_and_primal(::AbstractInnerEvaluator, ::Vec) = nothing

# derived methods that are used in other places
_can_derive_gradients(aie::AbstractInnerEvaluator, ::Val) = false
_can_derive_gradients(aie::AbstractInnerEvaluator, ::Val{0}) = true
_can_derive_hessian(aie::AbstractInnerEvaluator, ::Val) = false
_can_derive_hessian(aie::AbstractInnerEvaluator, ::Union{Val{0},Val{1}}) = true
function _can_derive_gradients(ev::AbstractInnerEvaluator)
    if _provides_gradients(ev)
        return true
    else
        return _provides_jacobian(ev)
    end
end
function provides_gradients(aie::AbstractInnerEvaluator)
    return (
        _can_derive_gradients(aie, _diff_order(aie)) ||
        _can_derive_gradients(aie)
    )
end

function _can_derive_jacobian(ev::AbstractInnerEvaluator)
    if _provides_jacobian(ev)
        return true
    else
        return _provides_gradients(ev)
    end
end
function provides_jacobian(aie::AbstractInnerEvaluator)
    return _can_derive_jacobian(aie)
    #=    return (
            _can_derive_gradients(aie, _diff_order(aie)) ||
            _can_derive_jacobian(aie)
        )=#
end

provides_hessians(aie::AbstractInnerEvaluator) = _provides_hessians(aie) || _can_derive_hessian(aie, _diff_order(aie))

function partial_jacobian(ev::AbstractInnerEvaluator, x::Vec; output_numbers)
    J = _partial_jacobian(ev, x; output_numbers)
    if isnothing(J)
        return _jacobian(ev, x)[output_numbers, :]
    else
        return J
    end
end

# helper
function _zero_grad(aie, x)
    return spzeros(eltype(x), length(x))
end

function _zero_jac(aie, x)
    return spzeros(eltype(x), num_outputs(aie), length(x))
end

function _zero_partial_jac(aie, x; output_numbers)
    return spzeros(eltype(x), length(output_numbers), length(x))
end

function _jacobian_from_grads(ev, x, ::Nothing)
    return transpose(
        reduce(
            hcat,
            gradient(ev, x; output_number) for output_number = 1:num_outputs(ev)
        )
    )
end

# (helper)
function _jacobian_from_grads(ev, x, output_numbers)
    return transpose(
        reduce(
            hcat,
            gradient(ev, x; output_number) for output_number in output_numbers
        )
    )
end

function jacobian(ev::T, x::Vec; output_numbers=nothing) where {T<:AbstractInnerEvaluator}
    if _provides_jacobian(ev)
        if isnothing(output_numbers)
            return _jacobian(ev, x,)
        else
            return partial_jacobian(ev, x; output_numbers)
        end
    elseif _provides_gradients(ev)
        return _jacobian_from_grads(ev, x, output_numbers)
    else
        if _can_derive_gradients(ev, _diff_order(ev))
            if isnothing(output_numbers)
                return _zero_jac(ev, x)
            else
                return _zero_partial_jac(ev, x; output_numbers)
            end
        end
        error("`jacobian` not availabe for evaluator of type $(T).")
    end
end

function gradient(ev::T, x::Vec; output_number) where {T<:AbstractInnerEvaluator}
    if _provides_gradients(ev)
        return _gradient(ev, x; output_number)
    elseif _provides_jacobian(ev)
        return vec(jacobian(ev, x; output_numbers=[output_number,]))
    else
        if _can_derive_gradients(ev, _diff_order(ev))
            return _zero_grad(ev, x)
        end
        error("`gradient` not availabe for evaluator of type $(T).")
    end
end

function jacobian_and_primal(ev::AbstractInnerEvaluator, x::Vec)
    tmp = _jacobian_and_primal(ev, x)
    if isnothing(tmp)
        return jacobian(ev, x), eval_at_vec(ev, x)
    end
    return tmp
end

function hessian(ev::T, x::Vec; output_number) where {T<:AbstractInnerEvaluator}
    if _provides_hessians(ev)
        return _hessian(ev, x; output_number)
    else
        if _can_derive_hessian(ev, _diff_order(ev))
            n_vars = length(x)
            return spzeros(eltype(x), n_vars, n_vars)
        end
        error("`hessian` not availabe for evaluator of type $(T).")
    end
end


#=====================================================================
AbstractOuterEvaluator
=====================================================================#
# An `ov::AbstractOuterEvaluator` has one set or several sets of
# `input_indices(ov)`, referencing either variables or
# dependent variables (output indices of other evaluators).
# `ov` also has a **single** set of `output_indices(ov)` and
# it somehow transforms values associated with the concatenated
# input indices to result values for the output indices.
# More precisely, an `inner_transformer(ov)::AbstractInnerEvaluator` is specified,
# and  this inner evaluator is used to transfrom the value vectors.
# Alternatively, the method `_transform_input_dict` can be overwritten,
# but then custom derivatives have to be specified too.
# Each `ov::AbstractOuterEvaluator` is a node in an evaluator tree:
# It either is a root node (`is_atomic(ov) == true`) or the function
# `input_provider(ov,l)` gives some other `av::AbstractOuterEvaluator`
# such that `input_indices(ov,l) == output_indices(av)`.

# The AbstractOuterEvaluator interface also "forwards" the AbstractInnerEvaluator methods.
Base.broadcastable(ov::AbstractOuterEvaluator) = Ref(ov)

"Return `true` if the outer evaluator takes only variables as its input 
and evaluates a single `AbstractInnerEvaluator`."
is_atomic(::AbstractOuterEvaluator{F}) where {F} = F

input_indices(::AbstractOuterEvaluator{true}) = VariableIndex[]
output_indices(::AbstractOuterEvaluator) = DependentIndex[]

num_input_sets(::AbstractOuterEvaluator) = 1

input_indices(::AbstractOuterEvaluator{false}, i) = DependentIndex[]

function input_indices(ov::AbstractOuterEvaluator{false})
    return reduce(union, input_indices(ov, l) for l = 1:num_input_sets(ov))
end

# ( mandatory )
inner_transformer(::AbstractOuterEvaluator) = nothing
input_provider(::AbstractOuterEvaluator{false}, l)::AbstractOuterEvaluator = nothing

# defaults
input_provider(aoe::AbstractOuterEvaluator{true}, l) = aoe # don't change
model_cfg(::AbstractOuterEvaluator)::AbstractSurrogateConfig = DUMMY_CONFIG
max_evals(ov::AbstractOuterEvaluator) = max_evals(model_cfg(ov))


# derived
num_inputs(ov::AbstractOuterEvaluator{true}) = length(input_indices(ov))
num_outputs(ov::AbstractOuterEvaluator) = length(output_indices(ov))
num_inputs(ov::AbstractOuterEvaluator{false}, i) = length(input_indices(ov, i))
function num_inputs(ov::AbstractOuterEvaluator{false})
    return sum(num_inputs(ov, i) for i = 1:num_input_sets(ov))
end

function outer_evaluator_or_model(ov, models)
    haskey(models, ov) && return models[ov]
    return ov
end

function provides_jacobian(ov::AbstractOuterEvaluator{true}; models=nothing)
    return provides_jacobian(inner_transformer(outer_evaluator_or_model(ov, models)))
end
function provides_gradients(ov::AbstractOuterEvaluator{true}; models=nothing)
    return provides_gradients(inner_transformer(outer_evaluator_or_model(ov, models)))
end

function provides_jacobian(ov::AbstractOuterEvaluator{false}; models=nothing)
    return provides_jacobian(inner_transformer(outer_evaluator_or_model(ov, models)) &&
           all(provides_jacobian(input_provider(ov, l); models) for l = 1:num_input_sets(ov)))
end

function provides_gradients(ov::AbstractOuterEvaluator{false}; models=nothing)
    return provides_gradients(inner_transformer(outer_evaluator_or_model(ov, models)) &&
           all(provides_gradients(input_provider(ov, l); models) for l = 1:num_input_sets(ov)))
end

# (helper)
function _collect_input_vector(ov, xd)
    return collect(getindices(xd, input_indices(ov)))
end

# (helper)
function _transform_input_vector(ov, x; models=nothing)
    return Dictionary(
        output_indices(ov),
        eval_at_vec(inner_transformer(outer_evaluator_or_model(ov, models)), x)
    )
end

function _transform_input_dict(
    ov::AbstractOuterEvaluator,
    xd; # :: AbstractDictionary{<:ScalarIndex, <:Real};
    models=nothing
)
    x = _collect_input_vector(ov, xd)
    return _transform_input_vector(ov, x; models)
end

function eval_at_dict(
    ov::AbstractOuterEvaluator{true},
    xd; # :: AbstractDictionary{<:ScalarIndex,<:Real}
    models=nothing
)
    _transform_input_dict(ov, xd; models)
end

function input_variable_indices(ov::AbstractOuterEvaluator{true})
    return input_indices(ov)
end

function input_variable_indices(ov::AbstractOuterEvaluator{false})
    ind = VariableIndex[]
    for l = 1:num_input_sets(ov)
        union!(ind, input_variable_indices(input_provider(ov, l)))
    end

    return input_indices(ov)
end

# (helper)
# NOTE I am not entirely sure anymore why I wrote `overwrite`
# It is inspired by `merge` but I guess it keeps the input indices
# of `out` instead of copying which might make co-iteration faster (???)
function overwrite(_out::AbstractDictionary{K,T}, _src::AbstractDictionary{K,V}) where {K,V,T}
    X = Base.promote_type(V, T)
    out = map(X, _out)
    src = map(X, _src)
    return overwrite(out, src)
end

function overwrite(out::AbstractDictionary{K,V}, src::AbstractDictionary{K,V}) where {K,V}
    overwrite!(out, src)
    return out
end

function overwrite!(out::AbstractDictionary{K,V}, src::AbstractDictionary{K,V}) where {K,V}
    for (k, v) = pairs(src)
        setindex!(out, v, k)
    end
    return nothing
end

function overwrite!(out::AbstractDictionary{K,T}, src::AbstractDictionary{K,V}) where {K,T,V}
    for (k, v) = pairs(src)
        setindex!(out, convert(T, v), k)
    end
    return nothing
end

function _recursive_input_dict(ov, xd::AbstractDictionary{K,T}; models=nothing) where {K,T}
    N = num_input_sets(ov)
    out = similar(input_indices(ov), T)
    for l = 1:N
        iv = input_provider(ov, l)
        vals_dict = eval_at_dict(iv, xd; models)
        out = overwrite(out, vals_dict)
    end
    return out
end

function eval_at_dict(
    ov::AbstractOuterEvaluator{false},
    xd;
    models=nothing
)
    _xd = _recursive_input_dict(ov, xd)
    return _transform_input_dict(ov, _xd; models)
end

function eval_at_dict!(
    ov::AbstractOuterEvaluator{true},
    xd::AbstractDictionary{>:SCALAR_INDEX,<:Real};
    models=nothing
)
    out_ind = output_indices(ov)
    _recompute = any(!haskey(xd, oi) for oi in out_ind)
    if _recompute
        merge!(xd, _transform_input_dict(ov, xd; models))
    end
    return xd
end

function _recursive_input_dict!(ov, xd::AbstractDictionary; models=nothing)
    N = num_input_sets(ov)
    for l = 1:N
        iv = input_provider(ov, l)
        eval_at_dict!(iv, xd; models)
    end
    return xd
end

function eval_at_dict!(
    ov::AbstractOuterEvaluator{false},
    xd::AbstractDictionary{>:SCALAR_INDEX,<:Real};
    models=nothing
)

    out_ind = output_indices(ov)
    if !all(haskey(xd, k) for k = out_ind)
        ξd = _recursive_input_dict!(ov, xd; models)
        merge!(xd, _transform_input_dict(ov, ξd; models))
    end
    return xd
end

function Base.in(i, indices::AbstractIndices{I}) where {I}
    local ret::Bool
    try
        ret = convert(I, i) in indices
    catch
        ret = false
    end
    return ret
end

function _input_changed(ov :: AbstractOuterEvaluator{true}, src, trgt)
    for ii in input_indices(ov)
        if !haskey(trgt, ii)
            return true
        end
        if haskey(src, ii)
            if getindex(src, ii) != getindex(trgt, ii)
                return true
            end
        end
    end
    return false
end
function _input_changed(ov::AbstractOuterEvaluator{false}, src, trgt, old_inputs )
    for ii in input_indices(ov)
        old_val = getindex(old_inputs, ii)
        if isnan(old_val)
            return true
        end
        if haskey(trgt, ii)
            if getindex(trgt, ii) != old_val
                return true
            end
        end
    end
    return false
end

function _output_missing(ov, src, trgt)
    for oi in output_indices(ov)
        if !haskey(trgt, oi)
            return true
        end
    end
    return false
end

function make_target(xd::AbstractDictionary{<:ScalarIndex,V}) where {V}
    return Dictionary{SCALAR_INDEX,V}(copy(keys(xd)), fill(V(NaN), length(xd)))
end

function eval_into_target!(
    ov::AbstractOuterEvaluator{true},
    src::AbstractDictionary{<:ScalarIndex,<:Real},
    trgt::AbstractDictionary{>:SCALAR_INDEX,<:Real};
    models=nothing
)
    if _output_missing(ov, src, trgt) || _input_changed(ov, src, trgt)
        merge!(trgt, src) # update input indices
        merge!(trgt, _transform_input_dict(ov, trgt; models)) # update output
    end

    return trgt
end

function _eval_preceeding_into_target!(
    ov, src, trgt; models=nothing
    #ov, src, trgt, ip_indices = nothing ; models=nothing
)
    ip_ind = (1:num_input_sets(ov))
    #ip_ind = isnothing(ip_indices) ? (1:num_input_sets(ov)) : ip_indices
    for l = ip_ind
        ip = input_provider(ov, l)
        eval_into_target!(ip, src, trgt; models)
    end
    return trgt
end

function Base.show(io::IO, ov::T) where {T<:AbstractOuterEvaluator}
    print(io, Base.typename(T).name)
end

function eval_into_target!(
    ov::AbstractOuterEvaluator{false},
    src::AbstractDictionary{<:ScalarIndex,<:Real},
    trgt::AbstractDictionary{>:SCALAR_INDEX,F};
    models=nothing
) where {F<:Real}
    old_inputs = dictionary(
        ii => haskey(trgt, ii) ? getindex(trgt, ii) : F(NaN) for ii in input_indices(ov)
    )
    _eval_preceeding_into_target!(ov, src, trgt; models)

    if _output_missing(ov, src, trgt) || _input_changed(ov, src, trgt, old_inputs)
        merge!(trgt, _transform_input_dict(ov, trgt; models))
    end
    return trgt
end
#==============================#
# Forward Mode Differentiation
#==============================#

# (helper)
"""
    _jacobian_and_primal(ov, xd)

Given an evaluator `ov` and some dictionary `xd`,
return a Dictionary mapping the output indices
of `ov` to the jacobian values at `xd` and a
Dictionary mapping to the primal values.
"""
function _jacobian_and_primal(ov, xd; models=nothing)
    x = _collect_input_vector(ov, xd)
    J, y = jacobian_and_primal(inner_transformer(outer_evaluator_or_model(ov, models)), x)
    in_ind = input_indices(ov)
    out_ind = output_indices(ov)
    return (
        Dictionary(
            out_ind,
            [
                Dictionary(
                    in_ind,
                    Jrow
                )
                for Jrow = eachrow(J)# ! not copying, each Jcol is a `SubArray`
            ]
        ),
        Dictionary(out_ind, y)
    )
end

"""
    _jacobian_and_primal!(ov, xd)

Given an evaluator `ov` and some dictionary `xd`,
return a Dictionary mapping the output indices
of `ov` to the jacobian values at `xd`.
Also modify `xd` to contain all forward evaluation results.
"""
function _jacobian_and_primal!(ov, xd; models=nothing)

    out_ind = output_indices(ov)
    x = _collect_input_vector(ov, xd)
    if !all(haskey(xd, k) for k = out_ind)
        J, y = jacobian_and_primal(inner_transformer(outer_evaluator_or_model(ov, models)), x)
        merge!(xd, Dictionary(out_ind, y))
    else
        J = jacobian(inner_transformer(outer_evaluator_or_model(ov, models)), x)
    end

    in_ind = input_indices(ov)
    return Dictionary(
        out_ind,
        [
            Dictionary(
                in_ind,
                Jrow
            )
            for Jrow = eachrow(J)# ! not copying, each Jcol is a `SubArray`
        ]
    )
end

function _transposed_jacobian_and_primal(ov, xd; models=nothing)
    x = _collect_input_vector(ov, xd)
    J, y = jacobian_and_primal(inner_transformer(outer_evaluator_or_model(ov, models)), x)
    in_ind = input_indices(ov)
    out_ind = output_indices(ov)
    return (
        Dictionary(
            in_ind,
            [
                Dictionary(
                    out_ind,
                    Jcol
                )
                for Jcol = eachcol(J)# ! not copying, each Jcol is a `SubArray`
            ]
        ),
        Dictionary(out_ind, y)
    )
end

function _transposed_jacobian_and_primal!(ov, xd; models=nothing)

    out_ind = output_indices(ov)
    x = _collect_input_vector(ov, xd)
    if !all(haskey(xd, k) for k = out_ind)
        J, y = jacobian_and_primal(inner_transformer(outer_evaluator_or_model(ov, models)), x)
        merge!(xd, Dictionary(out_ind, y))
    else
        J = jacobian(inner_transformer(outer_evaluator_or_model(ov, models)), x)
    end

    in_ind = input_indices(ov)
    return Dictionary(
        in_ind,
        [
            Dictionary(
                out_ind,
                Jcol
            )
            for Jcol = eachcol(J)# ! not copying, each Jcol is a `SubArray`
        ]
    )
end

"""
    jacobian_and_primal(abstract_outer_evaluator, xd)

Return a `Dictionary` mapping the
`output_indices(abstract_outer_evaluator)` of type
`DependentIndex` to Dictionaries with keys that are subsets
of `keys(xd)`, such that each subdict describes the partial
derivative values of one output with respect to the variables
of `xd`.
Return also a primal evaluation Dictionary with output values
at `xd`.
"""
function jacobian_and_primal(ov::AbstractOuterEvaluator{true}, xd; models=nothing)
    return _jacobian_and_primal(ov, xd; models)
end

function jacobian_and_primal!(ov::AbstractOuterEvaluator{true}, xd; models=nothing)
    return _jacobian_and_primal!(ov, xd; models)
end

function transposed_jacobian_and_primal(ov::AbstractOuterEvaluator{true}, xd; models=nothing)
    return _transposed_jacobian_and_primal(ov, xd; models)
end

function transposed_jacobian_and_primal!(ov::AbstractOuterEvaluator{true}, xd; models=nothing)
    return _transposed_jacobian_and_primal!(ov, xd; models)
end

# (helper)
# for non-leaf evaluators with multiple input sets
# compute the (transposed) jacobian of the inputs and there values
# the returned dictionaries can then be used to compute the
# jacobian of `ov` at `xd` via the Chain rule.
function _transposed_jacobian_and_primal_of_inputs(ov, xd::AbstractDictionary{K,V}; models=nothing) where {K,V}
    in_ind = input_indices(ov) # these are the *output* indices of the input providers
    II = eltype(in_ind)

    jacT = Dictionary{VariableIndex,Dictionary{II,V}}()
    prim = Dictionary{II,V}()

    for l = 1:num_input_sets(ov)
        iv = input_provider(ov, l)
        # `iv` maps something to `input_indices(ov, l)`
        # all `input_indices(ov, ...)` are
        # disjoint by design and of type `DependentIndex`
        _jacT, _prim = transposed_jacobian_and_primal(iv, xd; models)

        jacT = _concat_sub_dicts(jacT, _jacT)
        prim = merge(prim, _prim)
    end
    return jacT, prim
end
function _transposed_jacobian_and_primal_of_inputs!(ov, xd::AbstractDictionary{K,V}; models=nothing) where {K,V}
    in_ind = input_indices(ov) # these are the *output* indices of the input providers
    II = eltype(in_ind)

    jacT = Dictionary{VariableIndex,Dictionary{II,V}}()
    for l = 1:num_input_sets(ov)
        iv = input_provider(ov, l)
        _jacT = transposed_jacobian_and_primal!(iv, xd; models)
        jacT = _concat_sub_dicts(jacT, _jacT)
    end
    return jacT
end

#=
function _jacobian_and_primal_of_inputs(ov, xd::AbstractDictionary{K,V}) where {K,V}
    in_ind = input_indices(ov) # these are the *output* indices of the input providers
    II = eltype(in_ind)

    jac = Dictionary{II,Dictionary{K,V}}()
    prim = Dictionary{II,V}()

    for l = 1:num_input_sets(ov)
        iv = input_provider(ov, l)
        _jac, _prim = jacobian_and_primal(iv, xd)

        jac = _concat_sub_dicts(jac, _jac)
        prim = merge(prim, _prim)
    end
    return jac, prim
end
=#

# (helper)
function _dot_prod(
    dictA::AbstractDictionary{K,T},
    dictB::AbstractDictionary{K,S}
) where {K,T,S}
    V = Base.promote_type(T, S)
    ret = V(0)
    for (ka, va) = pairs(dictA)
        if haskey(dictB, ka)
            ret += va * getindex(dictB, ka)
        end
    end
    return ret
end

function _mat_prod(dictA, dictB)
    return dictionary(
        out_ind => filter(
            !iszero,
            dictionary(
                in_ind => _dot_prod(lhs, rhs) for (in_ind, rhs) = pairs(dictB)
            )
        ) for (out_ind, lhs) = pairs(dictA)
    )
end

function jacobian_and_primal(ov::AbstractOuterEvaluator{false}, xd; models=nothing)
    RHS, _xd = _transposed_jacobian_and_primal_of_inputs(ov, xd; models)

    # `_xd` is the accumulated primal up to now
    # it is the input to the current node.
    # we calculate the jacobian and primal of this node:
    LHS, prim = _jacobian_and_primal(ov, _xd; models)

    # apply chain rule
    jac = _mat_prod(LHS, RHS)
    return jac, prim
end

function jacobian_and_primal!(ov::AbstractOuterEvaluator{false}, xd; models=nothing)
    RHS = _transposed_jacobian_and_primal_of_inputs!(ov, xd; models)

    # `_xd` is the accumulated primal up to now
    # it is the input to the current node.
    # we calculate the jacobian and primal of this node:
    LHS = _jacobian_and_primal!(ov, xd; models)

    # apply chain rule
    jac = _mat_prod(LHS, RHS)
    return jac
end

# transposing a matrix product:
# (LHS * RHS^T)^T = RHS * LHS^T
# so below, `LHS` is the old `RHS`
# and transposition switches (but this is encoded by the dict structure already)
function transposed_jacobian_and_primal(ov::AbstractOuterEvaluator{false}, xd; models=nothing)
    LHS, _xd = _transposed_jacobian_and_primal_of_inputs(ov, xd; models)
    RHS, prim = _jacobian_and_primal(ov, _xd; models)

    # apply chain rule
    jacT = _mat_prod(LHS, RHS)
    return jacT, prim
end
function transposed_jacobian_and_primal!(ov::AbstractOuterEvaluator{false}, xd; models=nothing)
    LHS = _transposed_jacobian_and_primal_of_inputs!(ov, xd; models)
    RHS = _jacobian_and_primal!(ov, xd; models)

    # apply chain rule
    jacT = _mat_prod(LHS, RHS)
    return jacT
end

"""
    jacobian(abstract_outer_evaluator, xd)

Return a `Dictionary` mapping the
`output_indices(abstract_outer_evaluator)` of type
`DependentIndex` to Dictionaries with keys that are subsets
of `keys(xd)`, such that each subdict describes the partial
derivative values of one output with respect to the variables
of `xd`.
"""
function jacobian_dict(ov::AbstractOuterEvaluator, xd; models=nothing)
    return first(jacobian_and_primal(ov, xd; models))
end

function _jacobian_matrix_from_dict(
    jac_dict,
    func_inds=DependentIndex[],
    var_inds=SCALAR_INDEX[]
)
    in_ind = isempty(var_inds) ? reduce(union, keys(sub_dict) for sub_dict in jac_dict) : var_inds
    out_ind = isempty(func_inds) ? collect(keys(jac_dict)) : func_inds

    I = Int[]
    J = Int[]
    V = (valtype(valtype(jac_dict)))[]
    for (i, oind) in enumerate(out_ind)
        if haskey(jac_dict, oind)
            grad_dict = getindex(jac_dict, oind)
            for (j, iind) = enumerate(in_ind)
                if haskey(grad_dict, iind)
                    push!(I, i)
                    push!(J, j)
                    push!(V, grad_dict[iind])
                end
            end
        end
    end
    return sparse(I, J, V, length(out_ind), length(in_ind))
end

function jacobian_matrix(ov::AbstractOuterEvaluator, xd, var_inds=VariableIndex[]; models=nothing)
    in_ind = isempty(var_inds) ? collect(keys(xd)) : var_inds
    jac_dict = jacobian_dict(ov, xd; models)
    return _jacobian_matrix_from_dict(jac_dict, output_indices(ov), in_ind)
end

# (helper)
function _concat_sub_dicts(
    _out::AbstractDictionary{K,<:AbstractDictionary{I,V}},
    _src::AbstractDictionary{K,<:AbstractDictionary{I,W}},
    args...
) where {K,I,V,W}
    X = Base.promote_type(V, W)
    out = map(Dictionary{I,X}, _out)
    src = map(Dictionary{I,X}, _src)
    return _concat_sub_dicts(out, src, args...)
end

# (helper)
"""
    _concat_sub_dicts(out, src)

Given two Dictionary-of-Dictionary's, modify and return `out`
so that it contains all the keys of `out` and `src`.
If a key is present in both Dictionaries, `out` will
map it to the `merge` of the corresponding sub-dicts.
*Think of this function as an “outer join”.*
"""
function _concat_sub_dicts(
    out::AbstractDictionary{K,<:AbstractDictionary{I,V}},
    src::AbstractDictionary{K,<:AbstractDictionary{I,V}},
) where {K,I,V}
    for (k, sub_dict) = pairs(src)
        if haskey(out, k)
            set!(out, k, merge(getindex(out, k), sub_dict))
        else
            insert!(out, k, sub_dict)
        end
    end
    return out
end

#%%
include("_differentiation.jl")

#=====================================================================
WrappedUserFunc <: AbstractInnerEvaluator 
=====================================================================#
struct WrappedUserFunc{
    #	F <: Function, 
    B,
    D<:Union{Nothing,FuncContainerBackend}
} <: AbstractInnerEvaluator
    #func :: F
    func::Function
    num_outputs::Int
    differentiator::D
    counter::Base.RefValue{Int}
end

function WrappedUserFunc(func::F;
    num_outputs::Int,
    differentiator::D=nothing,
    can_batch::Bool=false
) where {
    F<:Function,
    D<:Union{Nothing,FuncContainerBackend},
}
    #return WrappedUserFunc{F, can_batch, D}(
    return WrappedUserFunc{can_batch,D}(
        func,
        num_outputs,
        differentiator,
        Ref(0)
    )
end

num_outputs(wuf::WrappedUserFunc) = wuf.num_outputs
num_eval_counter(wuf::WrappedUserFunc) = wuf.counter

_eval_at_vec(wuf::WrappedUserFunc, x::Vec) = wuf.func(x)

function _eval_at_vecs(wuf::WrappedUserFunc{true,<:Any}, X::VecVec)
    return wuf.func(X)
end

_provides_gradients(wuf::WrappedUserFunc) = true
_provides_jacobian(wuf::WrappedUserFunc) = true
_provides_hessians(wuf::WrappedUserFunc) = true

_provides_gradients(wuf::WrappedUserFunc{<:Any,Nothing}) = false
_provides_jacobian(wuf::WrappedUserFunc{<:Any,Nothing}) = false
_provides_hessians(wuf::WrappedUserFunc{<:Any,Nothing}) = false

function _gradient(wuf::WrappedUserFunc, x::Vec; output_number)
    return gradient(wuf.differentiator, wuf, x; output_number)
end

function _jacobian(wuf::WrappedUserFunc, x::Vec)
    return jacobian(wuf.differentiator, wuf, x)
end

function _partial_jacobian(wuf::WrappedUserFunc, x::Vec; output_numbers)
    return partial_jacobian(wuf.differentiator, wuf, x; output_numbers)
end

function _hessian(wuf::WrappedUserFunc, x::Vec; output_number)
    return hessian(wuf.differentiator, wuf, x; output_number)
end

#=====================================================================
InnerIdentity <: AbstractInnerEvaluator 
=====================================================================#
struct InnerIdentity <: AbstractInnerEvaluator
    n_out::Int
end

_diff_order(::InnerIdentity) = Val(1)

function _eval_at_vec(ii::InnerIdentity, x::Vec)
    @assert length(x) == ii.n_out
    return x
end

num_outputs(ii::InnerIdentity) = ii.n_out

_provides_gradients(::InnerIdentity) = true
_provides_jacobian(::InnerIdentity) = true
_provides_hessians(::InnerIdentity) = true

function _gradient(::InnerIdentity, x::Vec, output_number)
    return sparsevec([output_number,], true, length(x))
end

function _jacobian(::InnerIdentity, x::Vec)
    #return LinearAlgebra.I( num_outputs(ii) )
    return LinearAlgebra.I(length(x))
end

function _partial_jacobian(::InnerIdentity, x::Vec, output_numbers)
    m = length(output_numbers)
    n = length(x)
    return sparse(1:m, output_numbers, ones(Bool, m), m, n)
end

function _hessian(ii::InnerIdentity, args...)
    n = num_outputs(ii)
    return spzeros(Bool, n, n)
end

#=====================================================================
LinearVectorMapping <: AbstractInnerEvaluator 
=====================================================================#
@with_kw struct LinearVectorMapping{
    R<:Real,
    AT<:AbstractMatrix{R},
    BT<:AbstractVector{R},
} <: AbstractInnerEvaluator
    A::AT
    b::BT
    counter::Union{Base.RefValue{Int},EmptyRefCounter} = EMPTY_COUNTER #Ref(0)
    n_out::Int = size(A, 1)
    n_vars::Int = size(A, 2)

    function LinearVectorMapping(
        A::AT, b::BT, counter, n_out, n_vars
    ) where {R<:Real,AT<:AbstractMatrix{R},BT<:AbstractVector{R}}
        @assert n_out == length(b)
        return new{R,AT,BT}(A, b, counter, n_out, n_vars)
    end
end

_diff_order(::LinearVectorMapping) = Val(1)
function LinearVectorMapping(_A::AbstractMatrix{T}, _b::AbstractVector{F}) where {T,F}
    R = Base.promote_type(T, F)
    return LinearVectorMapping(;
        A=sparse_or_static(_A, R),
        b=sparse_or_static(_b, R)
    )
end

num_eval_counter(lvm::LinearVectorMapping) = lvm.counter
num_outputs(lvm::LinearVectorMapping) = lvm.num_outputs

function _eval_at_vec(lvm::LinearVectorMapping, x::Vec)
    return lvm.A * x .+ lvm.b
end

function _eval_at_vecs(lvm::LinearVectorMapping, X::VecVec)
    _X = reduce(hcat, X)    # can this be done lazily? ApplyArray(hcat, X...), but splatting is slow
    _tmp = lvm.A * _X .+ lvm.b
    return collect(eachcol(_tmp))
end

_provides_gradients(::LinearVectorMapping) = true
_provides_jacobian(::LinearVectorMapping) = true
_provides_hessian(::LinearVectorMapping) = true

function _gradient(lvm::LinearVectorMapping, x::Vec; output_number)
    return vec(lvm.A[output_number, :])
end
function _jacobian(lvm::LinearVectorMapping, x::Vec)
    return lvm.A
end
function _partial_jacobian(lvm::LinearVectorMapping, ::Vec; output_numbers)
    return lvm.A[output_numbers, :]
end
function _hessian(lvm::LinearVectorMapping, ::Vec; output_number)
    return spzeros(lvm.n_vars, lvm.n_vars)
end

#=====================================================================
OuterIdentity <: AbstractOuterEvaluator{true}
=====================================================================#
# ("atomic" building block)
Base.@kwdef struct OuterIdentity <: AbstractOuterEvaluator{true}
    input_indices::Vector{VariableIndex}
    num_inputs::Int = length(input_indices)
    inner::InnerIdentity = InnerIdentity(num_inputs)
    output_indices = Vector([DependentIndex() for i = 1:num_inputs])
end

inner_transformer(oi::OuterIdentity) = oi.inner
input_indices(oi::OuterIdentity) = oi.input_indices
output_indices(oi::OuterIdentity) = oi.output_indices
num_inputs(oi::OuterIdentity) = oi.num_inputs
num_outputs(oi::OuterIdentity) = num_inputs(oi)
model_cfg(::OuterIdentity) = DUMMY_CONFIG

#=====================================================================
VecFun <: AbstractOuterEvaluator{true}
=====================================================================#
Base.@kwdef struct VecFunc{
    I<:AbstractInnerEvaluator,
    C<:AbstractSurrogateConfig,
    #OI <: AbstractVector{<:DependentIndex}
} <: AbstractOuterEvaluator{true}
    transformer::I
    model_cfg::C = DUMMY_CONFIG
    num_outputs::Int = num_outputs(transformer)

    input_indices::Vector{VariableIndex}
    output_indices = [DependentIndex() for i = 1:num_outputs]

    num_inputs::Int = length(input_indices)
end

inner_transformer(vfun::VecFunc) = vfun.transformer

num_inputs(vfun::VecFunc) = vfun.num_inputs
num_outputs(vfun::VecFunc) = vfun.num_outputs
input_indices(vfun::VecFunc) = vfun.input_indices
output_indices(vfun::VecFunc) = vfun.output_indices
model_cfg(vfun::VecFunc) = vfun.model_cfg

#=====================================================================
ForwardingOuterEvaluator <: AbstractOuterEvaluator
=====================================================================#
struct ForwardingOuterEvaluator{A,VF<:AbstractOuterEvaluator{A}} <: AbstractOuterEvaluator{A}
    inner::VF
end

@forward(
    ForwardingOuterEvaluator.inner,
    (
        is_atomic,
        input_indices,
        output_indices,
        num_input_sets,
        inner_transformer,
        input_provider,
        num_inputs,
        num_outputs,
        model_cfg,
        # we could (and probably should) also forward some costly
        # derived methods:
        eval_at_dict,
        jacobian_dict,
        jacobian_matrix,
        jacobian_and_primal,
        jacobian_and_primal!,
        transposed_jacobian_and_primal!,
        transposed_jacobian_and_primal
    )
)

#=====================================================================
ProductOuterEvaluator <: AbstractOuterEvaluator{false}
=====================================================================#
Base.@kwdef struct ProductOuterEvaluator{
    ILeft<:AbstractOuterEvaluator,
    IRight<:AbstractOuterEvaluator,
} <: AbstractOuterEvaluator{false}
    inner_left::ILeft
    inner_right::IRight

    input_indices_left::Vector{DependentIndex} = output_indices(inner_left)
    input_indices_right::Vector{DependentIndex} = output_indices(inner_right)

    input_indices::Vector{DependentIndex} = union(input_indices_left, input_indices_right)

    num_inputs_left::Int = length(input_indices_left)
    num_inputs_right::Int = length(input_indices_right)

    num_outputs::Int = num_inputs_left + num_inputs_right

    output_indices::Vector{DependentIndex} = [DependentIndex() for i = 1:num_outputs]

    transformer::InnerIdentity = InnerIdentity(num_outputs)
end

inner_transformer(vfun::ProductOuterEvaluator) = vfun.transformer

num_outputs(vfun::ProductOuterEvaluator) = vfun.num_outputs
num_inputs(vfun::ProductOuterEvaluator) = vfun.num_inputs_left + vfun.num_inputs_right
num_input_sets(vfun::ProductOuterEvaluator) = 2
num_inputs(vfun::ProductOuterEvaluator, ::Val{1}) = vfun.num_inputs_left
num_inputs(vfun::ProductOuterEvaluator, ::Val{2}) = vfun.num_inputs_right
num_inputs(vfun::ProductOuterEvaluator, l) = num_inputs(vfun, Val(l))
output_indices(vfun::ProductOuterEvaluator) = vfun.output_indices
input_indices(vfun::ProductOuterEvaluator, ::Val{1}) = vfun.input_indices_left
input_indices(vfun::ProductOuterEvaluator, ::Val{2}) = vfun.input_indices_right
input_indices(vfun::ProductOuterEvaluator, l) = input_indices(vfun, Val(l))
input_indices(vfun::ProductOuterEvaluator) = vfun.input_indices

input_provider(vfun::ProductOuterEvaluator, ::Val{1}) = vfun.inner_left
input_provider(vfun::ProductOuterEvaluator, ::Val{2}) = vfun.inner_right
input_provider(vfun::ProductOuterEvaluator, l) = input_provider(vfun, Val(l))

model_cfg(::ProductOuterEvaluator) = DUMMY_CONFIG

#=====================================================================
CompositeEvaluator <: AbstractOuterEvaluator{false}
=====================================================================#
Base.@kwdef struct CompositeEvaluator{
    IE<:AbstractInnerEvaluator,
    IP<:AbstractOuterEvaluator,
    MC<:AbstractSurrogateConfig
} <: AbstractOuterEvaluator{false}

    outer::IE

    input_provider::IP
    input_indices::Vector{DependentIndex} = output_indices(input_provider)

    num_outputs::Int = num_outputs(outer)
    num_inputs = length(input_indices)
    output_indices = [DependentIndex() for i = 1:num_outputs]

    model_cfg::MC = DUMMY_CONFIG
end

num_input_sets(::CompositeEvaluator) = 1
num_inputs(ce::CompositeEvaluator, args...) = ce.num_inputs
num_outputs(ce::CompositeEvaluator) = ce.num_outputs
inner_transformer(ce::CompositeEvaluator) = ce.outer
input_indices(ce::CompositeEvaluator, args...) = ce.input_indices
input_provider(ce::CompositeEvaluator, args...) = ce.input_provider
output_indices(ce::CompositeEvaluator) = ce.output_indices
model_cfg(ce::CompositeEvaluator) = ce.model_cfg


@with_kw struct CompositeOuterEvaluators{
    L<:AbstractOuterEvaluator,
    R<:AbstractOuterEvaluator
} <: AbstractOuterEvaluator{false}
    lhs::L
    rhs::R

    @assert isempty(setdiff(input_indices(lhs), output_indices(rhs)))
end

# I am not sure if we need these standard definitions...
num_input_sets(coe::CompositeOuterEvaluators) = num_input_sets(coe.rhs)
num_inputs(coe::CompositeOuterEvaluators) = num_inputs(coe.rhs)
num_outputs(coe::CompositeOuterEvaluators) = num_outputs(coe.rhs)
inner_transformer(coe::CompositeOuterEvaluators) = inner_transformer(coe.lhs)
input_indices(coe::CompositeOuterEvaluators, args...) = input_indices(coe.rhs, args...)
input_provider(coe::CompositeOuterEvaluators, args...) = input_provider(coe.rhs, args...)
output_indices(coe::CompositeOuterEvaluators) = input_indices(coe.lhs)
model_cfg(coe::CompositeOuterEvaluators) = model_cfg(coe.lhs)

# Instead I overwrite the most important methods:
function eval_at_dict(coe::CompositeOuterEvaluators, xd; models=nothing)
    rhs_res = eval_at_dict(coe.rhs, xd; models)
    return eval_at_dict(coe.lhs, rhs_res; models)
end
function eval_at_dict!(
    coe::CompositeOuterEvaluators,
    xd::AbstractDictionary{>:SCALAR_INDEX, <:Real};
    models = nothing
)
    eval_at_dict!(coe.rhs, xd; models)
    return eval_at_dict!(coe.lhs, xd; models)
end

function eval_into_target!(
    coe::CompositeOuterEvaluators,
    src::AbstractDictionary{<:ScalarIndex,<:Real},
    trgt::AbstractDictionary{>:SCALAR_INDEX,<:Real};
    models=nothing
)
    eval_into_target!(coe.rhs, src, trgt; models)
    return eval_into_target!(coe.lhs, trgt, trgt; models)
end
