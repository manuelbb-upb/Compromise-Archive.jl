# # Surrogate Modelling

# In our optimizer, surrogate models originate from a mathematical
# context.
# That is, we think of them as vector-valued functions to approximate
# other vector-valued functions.
# As such, it is fitting to have them implement the `AbstractInnerEvaluator`
# interface.
# Each model type should inherit from `AbstractSurrogateModel`.
abstract type AbstractSurrogateModel <: AbstractInnerEvaluator end

# To allow for customization of the models, use a type inheriting
# `AbstractSurrogateConfig`. Users can then pass configuration
# objects when adding functions to a problem.
abstract type AbstractSurrogateConfig end

# From these configuration objects the corresponding model type is
# determined with `_model_type`.
# Copy the below function and make it specific to your model type
# by changing `AbstractSurrogateConfig` and `AbstractSurrogateModel`
# appropriately.
function _model_type(
    cfg::AbstractSurrogateConfig,
    mop::AbstractMOP,
    x::AbstractDictionary{SCALAR_INDEX,F},
    Δ::F,
    database::DB{F},
    x_index::Int
) where {F<:AbstractFloat}
    return AbstractSurrogateModel
end

# Similarly, there is (internal) meta data.
# We employ a 2-phase construction process.
# In the first phase, the model construction functions are
# presented with the current database of evaluations
# and can prepare the construction by asking for additional
# evaluations.
# Information for the finalization of construction should
# be stored in objects of dedicated type inheriting
# `AbstractSurrogateMeta`.
abstract type AbstractSurrogateMeta end

# To use type stable storage containers, it would again
# be nice to have a function predicting the meta type:
function _meta_type(
    cfg::AbstractSurrogateConfig,
    mop::AbstractMOP,
    x::AbstractDictionary{SCALAR_INDEX,F},
    Δ::F,
    database::DB{F},
    x_index::Int
) where {F<:AbstractFloat}
    return AbstractSurrogateMeta
end

# ## `AbstractSurrogateConfig` Interface

# Methods to be implemented by each type inheriting from
# `AbstractSurrogateConfig` and their defaults:

"""
    combinable( cfg :: AbstractSurrogateConfig )

Return a `bool` indicating whether or not we want
to built a single model for functions with equal
configurations `cfg` by stacking them vertically.
"""
combinable(::AbstractSurrogateConfig)::Bool = false

"""
    needs_gradients( cfg :: AbstractSurrogateConfig )
Return a `bool` indicating if first order derivatives are
required for model construction.
As the default model is exact, we need gradients for the
descent step calculation.
"""
needs_gradients(::AbstractSurrogateConfig)::Bool = true

"""
    needs_hessians( cfg :: AbstractSurrogateConfig )
Return a `bool` indicating if second order derivatives are
required for model construction.
"""
needs_hessians(::AbstractSurrogateConfig)::Bool = false

"""
    log_meta(cfg::AbstractSurrogateConfig)::Bool
Return `true`, if the output of `_json_info(model,meta)`
in each iteration should be included in the
iteration information returned to the user.
"""
log_meta(cfg::AbstractSurrogateConfig)::Bool = false

# ## `AbstractSurrogateModel` Interface

"""
    fully_linear( mod :: AbstractSurrogateModel )
Return a `bool` indicating if the current model qualifies
as fully linear or not.
"""
fully_linear(mod::AbstractSurrogateModel) = true

# ## `AbstractSurrogateMeta` Interface
