# depends on
# UUIDs
# `LinearVectorMapping`
# `Indices`
# VariableIndex
# ScaledVariableIndex

abstract type VariableTransformation <: AbstractOuterEvaluator{true} end

function make_scaled(var_inds)
    return [ScaledVariableIndex(vi) for vi in var_inds]
end
function make_unscaled(scaled_var_inds)
    return [svi.variable_index for svi in scaled_var_inds]
end

Base.@kwdef struct AffineScaling{
    T<:Union{LinearVectorMapping,InnerIdentity}
} <: VariableTransformation
    id::Int = 1
    input_output_indices::Dictionary{VariableIndex,ScaledVariableIndex}
    num_vars::Int = length(input_output_indices)
    transformer::T = InnerIdentity(num_vars)
end

Base.@kwdef struct InverseAffineScaling{
    T<:Union{LinearVectorMapping,InnerIdentity}
} <: VariableTransformation
    id::Int = 1
    input_output_indices::Dictionary{ScaledVariableIndex,VariableIndex}
    num_vars::Int = length(input_output_indices)
    transformer::T = InnerIdentity(num_vars)
end

const SomeAffineScaling{T} = Union{AffineScaling{T},InverseAffineScaling{T}} where {T};

input_indices(scaler::SomeAffineScaling) = collect(keys(scaler.input_output_indices))
output_indices(scaler::SomeAffineScaling) = collect(scaler.input_output_indices)
inner_transformer(scaler::SomeAffineScaling) = scaler.transformer
model_cfg(::SomeAffineScaling) = DUMMY_CONFIG
num_outputs(scaler::SomeAffineScaling) = scaler.num_vars

function invert_transformer(scaler::SomeAffineScaling{T}) where {T<:InnerIdentity}
    return scaler.transformer
end

function invert_transformer(scaler::SomeAffineScaling{T}) where {T<:LinearVectorMapping}
    # ξ = A x + b
    # x = A⁻¹(ξ - b)
    A = LinearAlgebra.inv(scaler.transformer.A)
    b = -A * scaler.transformer.b
    return LinearVectorMapping(; A, b)
end

function invert(scaler::SomeAffineScaling)
    transformer = invert_transformer(scaler)
    return InverseAffineScaling(;
        id=scaler.id,
        input_output_indices=Dictionary(values(scaler.input_output_indices), keys(scaler.input_output_indices)),
        transformer
    )
end

function Base.indexin(target_indices::AbstractIndices, partial_indices::AbstractIndices)
    return (findfirst(isequal(ind), target_indices) for ind in partial_indices)
end
function partial_scaler_for_inputs(scaler::T, inds) where {T<:SomeAffineScaling}
    matrix_indices = indexin(input_indices(scaler), inds)
    return T(
        ;
        A=scaler.A[matrix_indices],
        b=scaler.b[matrix_indices],
        input_output_indices=getindices(scaler.input_output_indices, inds),
        id=scaler.id
    )
end

