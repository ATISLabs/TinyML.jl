mutable struct BitTensor{N} <: AbstractArray{Bool, N}
    chunks::Array{UInt64, N}
    len::Int
    dims::NTuple{N, Int}

    function BitTensor{1}(value, length::Int) 
        if value == true
            return trues(length)
        elseif value == false
            return falses(length)
        else
            return bitrand(length)
        end
    end

    function BitTensor{2}(value, rows::Int, cols::Int)
        #inverted to be cache friendly when using iterating over rows
        chunkDims = (ceil(Int, cols / 64), rows)
        if value == true
            chunks = fill(typemax(UInt64), chunkDims)
        elseif value == false
            chunks = zeros(UInt64, chunkDims)
        else
            chunks = rand(UInt64, chunkDims)
        end

        if value != false
            rest = 64 - (cols % 64)
            if rest < 64
                for i in 1:rows
                    chunks[end,i] = (chunks[end,i] << rest) >> rest
                end
            end
        end

        len = rows * cols
        bmat = new(chunks, len)
        bmat.dims = (rows, cols)

        return bmat
    end

    function BitTensor{N}(value, dims::Vararg{Int, N}) where {N}
        chunkDims = [dim for dim in dims]
        chunkDims[1] = ceil(Int, dims[2] / 64)
        chunkDims[2] = dims[1]
        chunkDims = Tuple(chunkDims)

        if value == true
            chunks = fill(typemax(UInt64), chunkDims)
        elseif value == false
            chunks = zeros(UInt64, chunkDims)
        else
            chunks = rand(UInt64, chunkDims)
        end

        if value != false
            rest = 64 - (dims[2] % 64)
            if rest < 64
                counters = ones(Int, length(chunkDims) - 2)
                limits = chunkDims[3:end]
                llim = chunkDims[end]+1

                while llim > counters[end]
                    for i in 1:chunkDims[2]
                        chunks[end,i,counters...] = (chunks[end,i,counters...] << rest) >> rest
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

        len = 1
        for dim in dims
            len *= dim
        end
        bmat = new(chunks, len)
        bmat.dims = dims

        return bmat
    end
end

@inline Base.size(mat::BitTensor) = mat.dims
@inline Base.size(mat::BitTensor, dim::Int) = mat.dims[dim]

@inline Base.length(mat::BitTensor) = mat.len

@inline function Base.getindex(mat::BitTensor{2}, i::Int, j::Int)
    jMod = (j - 1)  % 64
    return Bool((mat.chunks[ceil(Int, j / 64), i] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BitTensor{3}, i::Int, j::Int, k::Int)
    jMod = (j - 1) % 64
    return Bool((mat.chunks[ceil(Int, j / 64), i, k] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BitTensor{4}, i::Int, j::Int, k::Int, l::Int)
    jMod = (j - 1) % 64
    return Bool((mat.chunks[ceil(Int, j / 64), i, k, l] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BitTensor, i::Int, j::Int, lastDims::Vararg{Int, N}) where {N}
    jMod = (j - 1) % 64
    return Bool((mat.chunks[ceil(Int, j / 64), i, lastDims...] & (1 << jMod)) >> jMod)
end




@inline function Base.setindex!(mat::BitTensor{2}, value::Bool, i::Int, j::Int)
    jMod = (j - 1) % 64
    mat.chunks[ceil(Int, j / 64), i] &= ~(1 << jMod)
    mat.chunks[ceil(Int, j / 64), i] |= value << jMod 
end
@inline function Base.setindex!(mat::BitTensor{3}, value::Bool, i::Int, j::Int, k::Int)
    jMod = (j - 1) % 64
    mat.chunks[ceil(Int, j / 64), i, k] &= ~(1 << jMod)
    mat.chunks[ceil(Int, j / 64), i, k] |= value << jMod 
end
@inline function Base.setindex!(mat::BitTensor{4}, value::Bool, i::Int, j::Int, k::Int, l::Int)
    jMod = (j - 1) % 64
    mat.chunks[ceil(Int, j / 64), i, k, l] &= ~(1 << jMod)
    mat.chunks[ceil(Int, j / 64), i, k, l] |= value << jMod 
end
@inline function Base.setindex!(mat::BitTensor, value::Bool, i::Int, j::Int, 
        lastDims::Vararg{Int, N}) where{N}
    jMod = (j - 1) % 64
    mat.chunks[ceil(Int, j / 64), i, lastDims...] &= ~(1 << jMod)
    mat.chunks[ceil(Int, j / 64), i, lastDims...] |= value << jMod 
end