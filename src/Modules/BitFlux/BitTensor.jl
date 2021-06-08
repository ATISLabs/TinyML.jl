struct BitTensor{N} <: AbstractArray{Bool, N}
    chunks::Array{UInt64, N}
    len::Int
    dims::NTuple{N, Int}
end

function BitTensor{1}(defaultvalue, length::Int) 
    if defaultvalue == true
        trues(length)
    elseif defaultvalue == false
        falses(length)
    else
        bitrand(length)
    end
end

function BitTensor{N}(defaultvalue, 
        rows::Int, cols::Int, dims::Vararg{Int, Y}) where {N, Y}
    @assert N == (Y+2) "Dimension mismatch: BitTensor{$(N)}, but passed $(Y+2) dimension lengths"

    chunk_rows = ceil(Int, cols / 64)
    chunksdims = (chunk_rows, rows, dims...) #cols and rows are swapped because
                                             #when iterating BitTensor over rows
                                             #chunks will be first indexed over columns.
                                             #Since Julia aligns arrays memory by columns, 
                                             #this swap improves performance for
                                             #applications that iterate by "row then col" order.

    chunks = initchunks(chunksdims, defaultvalue)
    if defaultvalue != false
        clearexcess(chunks)
    end
    len = reduce(*, chunksdims)

    BitTensor{N}(chunks, len, (rows, cols, dims...))
end

function initchunks(dims::NTuple, default)
    if default == true
        chunks = fill(typemax(UInt64), dims)
    elseif default == false
        chunks = zeros(UInt64, dims)
    else
        chunks = rand(UInt64, dims)
    end
end

function clearexcess(chunks::Array{UInt64, 2})
    rest = 64 - (size(chunks, 2) % 64)
    for i in 1:size(chunks, 2)
        chunks[end, i] = clear_rest(chunks[end, i], 
                                    rest)
    end
end

function clearexcess(chunks::Array{UInt64, N}) where {N}
    rest = 64 - (size(chunks, 2) % 64)
    if rest < 64
        counters = ones(Int, ndims(chunks) - 2)
        limits =  size(chunks)[3:end]
        lastlimit = limits[end]+1

        while lastlimit > counters[end]
            for i in 1:size(chunks, 2)
                chunks[end, i, counters...] = clear_rest(chunks[end, i, counters...], 
                                                        rest)
            end
            counters[1] += 1
            for i in 1:(length(counters)-1)
                if counters[i] > limits[i]
                    counters[i] = 1
                    counters[i+1] += 1
                end
            end
        end
    end
end

@inline clear_rest(val::UInt64, rest::Int) =  (val << rest) >> rest

@inline function Base.getindex(mat::BitTensor{2}, i::Int, j::Int)
    jMod = (j - 1)  % 64
    return Bool((mat.chunks[ceil(Int, j / 64), i] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BitTensor, i::Int, j::Int, 
        lastDims::Vararg{Int, N}) where {N}
    jMod = (j - 1) % 64
    return Bool((mat.chunks[ceil(Int, j / 64), i, lastDims...] & (1 << jMod)) >> jMod)
end

@inline function Base.setindex!(mat::BitTensor{2}, value::Bool, i::Int, j::Int)
    jMod = (j - 1) % 64
    mat.chunks[ceil(Int, j / 64), i] &= ~(1 << jMod)
    mat.chunks[ceil(Int, j / 64), i] |= value << jMod 
end
@inline function Base.setindex!(mat::BitTensor, value::Bool, i::Int, j::Int, 
        lastDims::Vararg{Int, N}) where{N}
    jMod = (j - 1) % 64
    mat.chunks[ceil(Int, j / 64), i, lastDims...] &= ~(1 << jMod)
    mat.chunks[ceil(Int, j / 64), i, lastDims...] |= value << jMod 
end

#= Displays =# 

# Getters and setters
@inline chunks(tensor::Union{BitTensor, BitArray}) = tensor.chunks
@inline Base.size(mat::BitTensor) = mat.dims
@inline Base.size(mat::BitTensor, dim::Int) = mat.dims[dim]
@inline Base.length(mat::BitTensor) = mat.len