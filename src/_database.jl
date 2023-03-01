struct DBResult{F<:AbstractFloat}
    result :: Dictionary{SCALAR_INDEX, F}
    scaling_id :: Base.RefValue{Int}
end

Base.@kwdef struct DB{F<:AbstractFloat}
    entries :: Dictionary{Int, DBResult{F}}
    next_id :: Base.RefValue{Int} = Ref(1)
end

Base.isempty(database::DB) = isempty(database.entries)
function empty!(database::DB)
    empty!(database.entries)
    database.next_id[] = 1
    return nothing
end

function Base.push!( database :: DB, res :: DBResult )
    id = database.next_id[]
    insert!(database.entries, id, res)
    database.next_id[] += 1
    return id
end

function db_request_vectors!(database :: DB, vecs :: VecVec)
    num_new = length(vecs)
    next_id = database.next_id[]
    new_ids = collect( next_id:(num_new-1) )
    database.next_id[] += num_new
    return Dictionary(new_ids, vecs)
end

Base.@kwdef struct LazyDBIterator{D}
    database :: D
    var_indices :: Vector{VariableIndex}
    scaled_var_indices :: Vector{ScaledVariableIndex} = Compromise.make_scaled(var_indices)
    output_indices :: Vector{DependentIndex}
end

function vectorize( res :: DBResult ) end

function Base.iterate(db_it :: LazyDBIterator)
    if isempty(db_it.database)
        return nothing
    end

end
