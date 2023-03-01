include("models/SimpleTaylor.jl")

# The model construction methods all expect an `AbstractOuterEvaluator` as input.
# It stores input and output dimension information.
# What is modelled is the `inner_transformer`.
# But what to do if we want to model multiple `AbstractOuterEvaluator`s that share
# the same `model_cfg`?
# We can gather these outer evaluators and concatenate their respective inner transformers.
# We then receive a meta model for the concatenated transformers.
# Models are injected into the evaluation process of outer evaluators by passing a
# dictionary mapping inner transformers to models (if applicable).
# Hence, we also should construct an `AbstractInnerEvaluator` type that takes
# the model for concatenated evaluators and destructures it.

#================================================
# GroupedInnerEvaluators
================================================#

Base.@kwdef struct GroupedInnerEvaluators{
    T<:AbstractVector{<:AbstractInnerEvaluator}
} <: AbstractInnerEvaluator
    evaluators::T

    output_dimensions::Vector{Int} = [Compromise.num_outputs(ev) for ev in evaluators]
    num_out::Int = sum(output_dimensions)

    # map an output index of the `GroupedInnerEvaluators` to
    # an index of an inner evaluator in `evaluators`
    # and its output index
    output_to_evaluator_output_mapping::Dictionary{Int,Tuple{Int,Int}} = begin
        d = Dictionary{Int,Tuple{Int,Int}}()
        l = 1
        for (i, ev) = enumerate(evaluators)
            for t = 1:Compromise.num_outputs(ev)
                set!(d, l, (i, t))
                l += 1
            end
        end
        d
    end
end

num_outputs(gie::GroupedInnerEvaluators) = gie.num_out

function _eval_at_vec(gie::GroupedInnerEvaluators, x::Vec)
    return reduce(
        vcat,
        _eval_at_vec(ev, x) for ev in gie.evaluators
    )
end

function _eval_at_vec(gie::GroupedInnerEvaluators, x::Vec, output_number)
    return reduce(
        vcat,
        let (i, l) = gie.output_to_evaluator_output_mapping[out_num]
            _eval_at_vec(gie.evaluators[i], x, l)
        end for out_num in output_number
    )
    # NOTE the above is not really effective, consider e.g. (1,1) and (1,2) -> seperate calls to same evaluator
    # Luckily, none of the model construction methods should want to evaluate
    # only specific outputs...
end

function _eval_at_vecs(gie::GroupedInnerEvaluators, X::VecVec)
    # by calling `eval_at_vecs` for each inner evaluator seperately,
    # we ensure that exploitation of parallelisation by inner evaluators
    # is respected
    res = _eval_at_vecs(first(gie.evaluators), X)
    for ev in gie.evaluators[2:end]
        tmp = _eval_at_vecs(ev, X)
        res = [vcat(r, t) for (r, t) = zip(res, tmp)]
    end
    return res
end

_diff_order(gie::GroupedInnerEvaluators) = maximum(_diff_order(ev) for ev in gie.evaluators)
_provides_gradients(gie::GroupedInnerEvaluators) = all(provides_gradients(ev) for ev in gie.evaluators)
_provides_jacobian(gie::GroupedInnerEvaluators) = all(provides_jacobian(ev) for ev in gie.evaluators)
_provides_hessians(gie::GroupedInnerEvaluators) = all(provides_hessians(ev) for ev in gie.evaluators)

function _jacobian(gie::GroupedInnerEvaluators, x::Vec)
    return reduce(
        vcat,
        jacobian(ev, x) for ev in gie.evaluators
    )
end

# NOTE Partial jacobians should not be needed and suffer
# from the same drawbacks `eval_at_vec(gie,x,output_number)`
# Gradients probably won't be needed either, but they are easy:
function _gradient(gie::GroupedInnerEvaluators, x::Vec; output_number)
    i, t = gie.output_to_evaluator_output_mapping[output_number]
    return gradient(gie.evaluators[i], x; output_number=t)
end
function _hessian(gie::GroupedInnerEvaluators, x::Vec; output_number)
    i, t = gie.output_to_evaluator_output_mapping[output_number]
    return hessian(gie.evaluators[i], x; output_number=t)
end

#================================================
# GroupedOuterEvaluators
================================================#
Base.@kwdef struct GroupedOuterEvaluators{
    E<:AbstractVector{<:AbstractOuterEvaluator},
    C<:AbstractSurrogateConfig,
    G<:GroupedInnerEvaluators,
} <: AbstractOuterEvaluator{false}
    evaluators::E

    model_cfg::C = Compromise.model_cfg(first(evaluators))

    num_inputs::Int = sum(Compromise.num_inputs(ev) for ev in evaluators)
    num_outputs::Int = sum(Compromise.num_outputs(ev) for ev in evaluators)
    num_input_sets::Int = sum(Compromise.num_input_sets(ev) for ev in evaluators)

    # NOTE we should not really use these output indices ever, so I guess we could
    # simply not generate them?
    output_indices::Vector{DependentIndex} = [DependentIndex() for i = 1:num_outputs]

    # map input set index of `GroupedOuterEvaluators` to
    # index of evaluator in `evaluators` and index of
    # input set for this evaluator
    input_set_mapping::Dictionary{Int,Tuple{Int,Int}} = begin
        d = Dictionary{Int,Tuple{Int,Int}}()
        l = 1
        for (i, ev) = enumerate(evaluators)
            for t = 1:Compromise.num_input_sets(ev)
                set!(d, l, (i, t))
                l += 1
            end
        end
        d
    end
    transformer::G = GroupedInnerEvaluators([Compromise.inner_transformer(ev) for ev in evaluators])
end

num_input_sets(goe::GroupedOuterEvaluators) = goe.num_input_sets
output_indices(goe::GroupedOuterEvaluators) = goe.output_indices
num_inputs(goe::GroupedOuterEvaluators) = goe.num_inputs
num_outputs(goe::GroupedOuterEvaluators) = goe.num_outputs
model_cfg(goe::GroupedOuterEvaluators) = goe.model_cfg
function input_provider(goe::GroupedOuterEvaluators, l)
    i, t = goe.input_set_mapping[l]
    return input_provider(goe.evaluators[i], t)
end
function input_indices(goe::GroupedOuterEvaluators, l)
    i, t = goe.input_set_mapping[l]
    return input_provider(goe.evaluators[i], t)
end
inner_transformer(goe::GroupedOuterEvaluators) = goe.transformer

#===============================================
# PartialInnerEvaluator
===============================================#

Base.@kwdef struct PartialInnerEvaluator{
    R<:Base.RefValue{<:AbstractInnerEvaluator}
} <: AbstractInnerEvaluator
    inner_ref::R
    output_numbers::Vector{Int}
    num_outputs::Int=length(output_numbers)
    provides_gradients::Bool = Compromise.provides_gradients(inner_ref[])
    provides_jacobian::Bool = Compromise.provides_jacobian(inner_ref[])
    provides_hessians::Bool = Compromise.provides_hessians(inner_ref[])
end

num_outputs(pie::PartialInnerEvaluator) = pie.num_outputs
_provides_gradients(pie::PartialInnerEvaluator) = pie.provides_gradients
_provides_jacobian(pie::PartialInnerEvaluator) = pie.provides_jacobian
_provides_hessians(pie::PartialInnerEvaluator) = pie.provides_hessians

function _eval_at_vec(pie::PartialInnerEvaluator, x::Vec)
    return eval_at_vec(pie.inner_ref[], x, pie.output_numbers)
end
function _eval_at_vecs(pie::PartialInnerEvaluator, X::VecVec)
    return eval_at_vecs(pie.inner_ref[], X, pie.output_numbers)
end

function _eval_at_vec(pie::PartialInnerEvaluator, x::Vec, l)
    return eval_at_vec(pie.inner_ref[], x, pie.output_numbers[l])
end
function _eval_at_vecs(pie::PartialInnerEvaluator, X::VecVec, l)
    return eval_at_vecs(pie.inner_ref[], X, pie.output_numbers[l])
end

function _jacobian(pie::PartialInnerEvaluator, x::Vec)
    return jacobian(pie.inner_ref[], x; output_numbers = pie.output_numbers)
end

function _gradient(pie::PartialInnerEvaluator, x::Vec; output_number)
    return gradient(pie.inner_ref[], x; output_number=pie.output_numbers[output_number])
end

function _hessian(pie::PartialInnerEvaluator, x::Vec; output_number)
    return hessian(pie.inner_ref[], x; output_number = pie.output_numbers[output_number])
end

function destructure_model(
    model::AbstractInnerEvaluator,
    evaluators::AbstractVector{<:AbstractInnerEvaluator},
    target_index::Int
)
    offset = if target_index == 1
        0
    else
        sum(num_outputs(ev) for ev in evaluators[1:target_index-1])
    end
    return PartialInnerEvaluator(
        ;inner_ref = Ref(model),
        output_numbers = offset .+ (1:num_outputs(evaluators[target_index]))
    )
end

#========================================================================================
# Functions to gather the Evaluators that need modelling
========================================================================================#

# With grouping:
#================#

function _models_by_type(aoe::AbstractOuterEvaluator{true}, mods_by_type)
    cfg = model_cfg(aoe)
    #if cfg != DUMMY_CONFIG
        model_set = get!(mods_by_type, cfg, Set())
#    push!(model_set, inner_transformer(aoe))
        push!(model_set, aoe)
    #end
    return nothing
end

function _models_by_type(aoe::AbstractOuterEvaluator{false}, mods_by_type)
    for l = num_input_sets(aoe)
        _models_by_type(input_provider(aoe, l), mods_by_type)
    end
    return nothing
end

function collect_models_by_type(mop::AbstractMOP)
    models_by_type = Dictionary{Any,Set{AbstractOuterEvaluator}}()
    for aoe in _all_functions(mop)
        _models_by_type(aoe, models_by_type)
    end
    return models_by_type
end

function generate_groups(models_by_type)
    outer_evaluators_with_models = AbstractOuterEvaluator[]

    for (cfg, aoe_set) in pairs(models_by_type)
        if combinable(cfg)
            push!(outer_evaluators_with_models, GroupedOuterEvaluators(
                ;evaluators=collect(aoe_set)
            ))
        else
            append!(outer_evaluators_with_models, aoe_set)
        end
    end
    return Tuple(outer_evaluators_with_models)
end

function outer_evaluators_needing_models(mop::AbstractMOP, do_groups::Val{true})
    models_by_type = collect_models_by_type(mop)
    return generate_groups(models_by_type)
end
# Without grouping:
#==================#

function _collect_evaluators(mop::AbstractMOP)
    s = Set{AbstractOuterEvaluator}()
    for aoe in _all_functions(mop)
        _collect_evaluators!(s, aoe)
    end
    return Tuple(s)
end

function _collect_evaluators!(s, aoe::AbstractOuterEvaluator{true})
    #if model_cfg(aoe) != DUMMY_CONFIG
        push!(s, aoe)
    #end
    nothing
end
function _collect_evaluators!(s, aoe::AbstractOuterEvaluator{false})
    for l=1:num_input_sets(aoe)
        _collect_evaluators!(s, input_provider(aoe, l))
    end
    nothing
end

function outer_evaluators_needing_models(mop::AbstractMOP, do_groups::Val{false})
    return _collect_evaluators(mop)
end

#========================================================================================
# Functions to construct models
========================================================================================#

function initialize_models(
    mop, x, Δ, database, problem_precision, x_index, evaluators_tuple
)
    # Union splitting is beginning to work well for large unions
    # We thus strongly type the dictionaries we use by gathering
    # return types beforehand.
    # This is only done once (during initialization).
    # Thereafter, these types are enforced througouht iterations.
    KEYS_TYPE = Union{typeof.(evaluators_tuple)...}

    METAS_TYPE = Union{(_meta_type(problem_precision, ev) for ev in tuple_of_outer_evaluators)...}

    MODS_TYPE = Union{(_model_type(problem_precision, ev) for ev in tuple_of_outer_evaluators)...}
end

function construct_models(
    mop :: AbstractMOP,
    x :: AbstractDictionary,
    Δ :: Real,
    database :: DB,
    x_index :: Int,
    problem_precision :: Type{<:AbstractFloat},
    evaluators_tuple :: Tuple{Vararg{<:AbstractOuterEvaluator}};
    model_dict :: Union{Nothing, AbstractDictionary} = nothing
)
    if isnothing(model_dict)
        return initialize_models(mop, x, Δ, database, x_index, problem_precision, evaluators_tuple)
    else 
        return update_models(mop, x, Δ, database, x_index, problem_precision, evaluators_tuple; model_dict)
    end

end

