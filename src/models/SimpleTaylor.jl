"Model for a single output of a MIMO function."
Base.@kwdef struct ScalarTaylorModel{F<:AbstractFloat}
    x0::F
    g::Vector{F}
    H::Union{Bool,Matrix{F}} = false
end

# Actual vector-valued-model:
struct SimpleTaylorModel{T<:ScalarTaylorModel} <: AbstractSurrogateModel
    scalar_models::Vector{T}
end

@with_kw struct SimpleTaylorConfig <: AbstractSurrogateConfig
    degree::UInt8 = UInt8(2)
    @assert 1 <= degree <= 2 "Can only build linear or quadratic models"
end

function _model_type(
    cfg::SimpleTaylorConfig,
    mop::AbstractMOP,
    x::AbstractDictionary{SCALAR_INDEX,F},
    Δ::F,
    database::DB{F},
    x_index::Int
) where {F<:AbstractFloat}
    return SimpleTaylorModel{F}
end

struct SimpleTaylorMeta <: AbstractSurrogateMeta end
function _meta_type(
    cfg::SimpleTaylorConfig,
    mop::AbstractMOP,
    x::AbstractDictionary{SCALAR_INDEX,F},
    Δ::F,
    database::DB{F},
    x_index::Int
) where {F<:AbstractFloat}
    return SimpleTaylorMeta
end

"""
    combinable( cfg :: SimpleTaylorConfig )

Return a `bool` indicating whether or not we want
to built a single model for functions with equal
configurations `cfg` by stacking them vertically.
"""
combinable(::SimpleTaylorConfig)::Bool = false

requires_database(::SimpleTaylorConfig)::Bool = false

"""
    needs_gradients( cfg :: SimpleTaylorConfig )
Return a `bool` indicating if first order derivatives are
required for model construction.
As the default model is exact, we need gradients for the
descent step calculation.
"""
needs_gradients(::SimpleTaylorConfig)::Bool = true

"""
    needs_hessians( cfg :: SimpleTaylorConfig )
Return a `bool` indicating if second order derivatives are
required for model construction.
"""
needs_hessians(cfg::SimpleTaylorConfig)::Bool = cfg.degree > 1

"""
    fully_linear( mod :: SimpleTaylorModel )
Return a `bool` indicating if the current model qualifies
as fully linear or not.
"""
fully_linear(mod::SimpleTaylorModel) = true

#%
function prepare_init_model(
    ev::AbstractOuterEvaluator,
    cfg::SimpleTaylorConfig,
    mop::AbstractMOP,
    scal::AbstractAffineScaler,
    x::AbstractDictionary{SCALAR_INDEX,F},
    Δ::F,
    database::DB{F},
    x_index::Int
) :: Tuple{SimpleTaylorMeta, Nothing} where {F<:AbstractFloat}
    return (SimpleTaylorMeta(),Nothing)
end

#%
function preparare_update_model(
    ev::AbstractOuterEvaluator,
    mod::SimpleTaylorModel,
    cfg::SimpleTaylorConfig,
    mop::AbstractMOP,
    scal::AbstractAffineScaler,
    x::AbstractDictionary{SCALAR_INDEX,F},
    Δ::F,
    database::DB{F},
    x_index::Int
) :: Tuple{SimpleTaylorMeta, Nothing} where {F<:AbstractFloat}
    return prepare_init_model(cfg,mop,scal,x,Δ,database,x_index)
end

# TODO move to `utilities.jl`
function extract_vec( xd :: AbstractDictionary, inds )
    return collect( getindices(xd, inds) )
end

function init_model(
    ev::AbstractOuterEvaluator,
    meta::SimpleTaylorMeta,
    cfg::SimpleTaylorConfig,
    mop::AbstractMOP,
    scal::AbstractAffineScaler,
    x::AbstractDictionary{SCALAR_INDEX,F},
    Δ::F,
    database::DB{F},
    x_index::Int
)::Tuple{SimpleTaylorModel,Nothing} where {F<:AbstractFloat}

    var_inds = _variable_indices(scal)
    scaled_var_inds = _scaled_variable_indices(scal)
    x0 = extract_vec(x0, var_inds)
    x0_unscaled = collect(getindices(x, _scaled_index.(_variable_indices(mop))))
end
