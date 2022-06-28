# utility functions that don't fit anywhere else and are used in several places
# todo use and improve `SimilarType.jl`

_mat_type( ::Type{<:AbstractMatrix}, :: Type{F} ) where F = Matrix{F}
_mat_type( T::Type{<:AbstractMatrix{F}}, :: Type{F} ) where F = T
_mat_type( :: Type{<:AbstractSparseMatrixCSC{<:Any, I}}, ::Type{F}) where {I,F} = SparseMatrixCSC{F,I}
_mat_type( T::Type{<:StaticMatrix}, ::Type{F} ) where F = similar_type(T, F)
_mat_type( :: T, F ) where T<:AbstractMatrix = _mat_type(T, F)

_vec_type( ::Type{<:AbstractVector}, :: Type{F} ) where F = Vector{F}
_vec_type( T::Type{<:AbstractVector{F}}, ::Type{F} ) where F = T
_vec_type( ::Type{<:AbstractSparseVector{<:Any,I}}, ::Type{F} ) where {I,F} = SparseVector{F,I}
_vec_type( T::Type{<:StaticVector}, ::Type{F} ) where F = similar_type(T, F)
_vec_type( x :: T, F ) where T<:AbstractVector = _vec_type(T, F)

_array_type(T::Type{<:AbstractVector}, F)=_vec_type(T,F)
_array_type(T::Type{<:AbstractMatrix}, F)=_mat_type(T,F)
_array_type(x :: T , F ) where T = _array_type(T,F)

function site_vec_to_dict(mop :: AbstractMOP, x :: Vec)
	return Dictionary( _variable_indices(mop), x )
end

function sparse_or_static( _A :: AbstractArray{T}, F :: Type{<:Number} ) where T
	len = length(A)
	density = sum(iszero.(_A)) / len
	A = if density <= .3
		F.(sparse(_A))
	else
		if len < 100
			sz = size(_A)
			SArray{Tuple{sz...}, F, length(sz)}(_A)
		else
			convert( _array_type( _A, F ), _A )
		end
	end
	return A
end