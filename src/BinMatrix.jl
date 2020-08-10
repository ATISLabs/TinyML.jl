"""
    mutable struct BinMatrix{N}

A type used for Bit operations. It is different than BitArray{2}
because it allocates ceil(columns / 64) for each line of the matrix.
In reason of that, it uses slightly more memory than BitArray, but
sometimes it increases performance by reducing shifting operations
depending of how it is used. 

# Structure
mutable struct BinMatrix
    chunks::Array{UInt64, 2}
    len::Int
    dims::NTuple{2, Int}
end

# Initialization
```julia
BinMatrix{N}(value [optional], lineNumber, columnNumber, ...)
```
Possible values
- undef     (random)
- true
- false

# Example
```julia_repl
julia> mat = BinMatrix{1}(undef, 5, 5)
julia> mat = BinMatrix{1}(true, 5, 5)
```
"""
mutable struct BinMatrix{N} <: AbstractArray{Bool, N}
    chunks::Array{UInt64, N}
    len::Int
    dims::NTuple{N, Int}

    function BinMatrix{N}(value, dims::Vararg{Int, N}) where {N}
        if length(dims) < 2
            error("A bit matrix must have at least two dimensions")
        else
            dims = dims

            len = 1
            for dim in dims
                len *= dim
            end

            chunkDims = [dim for dim in dims]
            chunkDims[2] = ceil(Int, chunkDims[2] / 64)
            chunkDims = Tuple(chunkDims)

            if value == undef
                chunks = rand(UInt64, chunkDims)
            elseif value == true
                chunks = fill(typemax(UInt64), chunkDims)
            else
                chunks = zeros(UInt64, chunkDims)
            end

            bmat = new(chunks, len)
            bmat.dims = dims

            return bmat
        end
    end
end

"""
    size(BinMatrix)

Returns the BinMatrix dimensions

# Example
```julia_repl
julia> size(BinMatrix(5, 6))
(5, 6)
```
"""
@inline Base.size(mat::BinMatrix) = mat.dims

"""
    length(BinMatrix)

Returns the number of elements in a BinMatrix

# Example
```julia_repl
julia> length(BinMatrix(2, 2))
4
```
"""
@inline Base.length(mat::BinMatrix) = mat.len

"""
    getindex(BinMatrix, line::Int, column::Int, ...)

Returns a Bool representing a bit in the specified
line and column of a BinMatrix

# Example
```julia_repl
julia> getindex(BinMatrix(2, 2), 1, 1)
true
julia> matrix[1, 1]
true
```
"""
@inline function Base.getindex(mat::BinMatrix{2}, i::Int, j::Int)
    jMod = j % 64
    return Bool((mat.chunks[i, ceil(Int, j / 64)] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BinMatrix{3}, i::Int, j::Int, k::Int)
    jMod = j % 64
    return Bool((mat.chunks[i, ceil(Int, j / 64), k] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BinMatrix{4}, i::Int, j::Int, k::Int, l::Int)
    jMod = j % 64
    return Bool((mat.chunks[i, ceil(Int, j / 64), k, l] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BinMatrix, i::Int, j::Int, lastDims::Vararg{Int, N}) where {N}
    jMod = j % 64
    return Bool((mat.chunks[i, ceil(Int, j / 64), lastDims...] & (1 << jMod)) >> jMod)
end

"""
    setBit(BinMatrix, line::Int, column::Int)

Sets a bit to 1 in the specified line and column
of a BinMatrix

# Example
```julia_repl
julia> setBit(BinMatrix(2, 2), 1, 1)
```
"""
@inline setBit(mat::BinMatrix, i::Int, j::Int, lastDims::Vararg{Int, N}) where {N} = 
    mat.chunks[i, ceil(Int, j / 64), lastDims...] |= 1 << (j % 64)

"""
    clearBit(BinMatrix, line::Int, column::Int)

Sets a bit to 0 in the specified line and column
of a BinMatrix

# Example
```julia_repl
julia> clearBit(BinMatrix(2, 2), 1, 1)
```
"""
@inline clearBit(mat::BinMatrix, i::Int, j::Int, lastDims::Vararg{Int, N}) where {N} =
    mat.chunks[i, ceil(Int, j / 64), lastDims...] &= ~(1 << (j % 64))

"""
    setindex!(BinMatrix, value::Bool, line::Int, column::Int)

Sets a bit to 'value' in the specified line and column
of a BinMatrix

# Example
```julia_repl
julia> setindex!(BinMatrix(2, 2), true, 1, 1)
true
julia> matrix[1, 1] = true
true
```
"""
@inline function Base.setindex!(mat::BinMatrix{2}, value::Bool, i::Int, j::Int)
    mat.chunks[i, ceil(Int, j / 64)] &= ~(1 << (j % 64))
    mat.chunks[i, ceil(Int, j / 64)] |= value << (j % 64) 
end
@inline function Base.setindex!(mat::BinMatrix{3}, value::Bool, i::Int, j::Int, k::Int)
    mat.chunks[i, ceil(Int, j / 64), k] &= ~(1 << (j % 64))
    mat.chunks[i, ceil(Int, j / 64), k] |= value << (j % 64) 
end
@inline function Base.setindex!(mat::BinMatrix{4}, value::Bool, i::Int, j::Int, k::Int, l::Int)
    mat.chunks[i, ceil(Int, j / 64), k, l] &= ~(1 << (j % 64))
    mat.chunks[i, ceil(Int, j / 64), k, l] |= value << (j % 64) 
end
@inline function Base.setindex!(mat::BinMatrix, value::Bool, i::Int, j::Int, lastDims::Vararg{Int, N}) where{N}
    mat.chunks[i, ceil(Int, j / 64), lastDims...] &= ~(1 << (j % 64))
    mat.chunks[i, ceil(Int, j / 64), lastDims...] |= value << (j % 64) 
end