# utility functions that don't fit anywhere else and are used in several places

_mat_type( ::Type{<:AbstractMatrix}, F ) = Matrix{F}
_mat_type( T::Type{<:AbstractMatrix{F}}, ::Type{F} ) where F = T
_mat_type( :: Type{<:AbstractSparseMatrixCSC{<:Any, I}}, F) where I = SparseMatrixCSC{F,I}
_mat_type( T::Type{<:StaticMatrix}, F ) = similar_type(T, F)
_mat_type( :: T, F ) where T<:AbstractMatrix = _mat_type(T, F)

_vec_type( ::Type{<:AbstractVector}, F ) = Vector{F}
_vec_type( T::Type{<:AbstractVector{F}}, ::Type{F} ) = T
_vec_type( ::Type{<:AbstractSparseVector{<:Any,I}}, F ) where I = SparseVector{F,I}
_vec_type( T::Type{<:StaticVector}, F ) = similar_type(T, F)
_vec_type( x :: T, F ) where T<:AbstractVector = _vec_type(T, F)

function site_vec_to_dict(mop :: AbstractMOP, x :: Vec)
	return Dictionary( _variable_indices(mop), x )
end