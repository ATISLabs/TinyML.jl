mutable struct BitTensor{N} <: AbstractArray{Bool, N}
    chunks::Array{UInt64, N}
    len::Int
    dims::NTuple{N, Int}

    function BitTensor{1}(value, length::Int) 
        if value == undef
            return bitrand(length)
        elseif value == true
            return trues(length)
        else
            return falses(length)
        end
    end

    function BitTensor{2}(value, rows::Int, cols::Int)
        chunkDims = (rows, ceil(Int, cols / 64))
        if value == undef
            chunks = rand(UInt64, chunkDims)
        elseif value == true
            chunks = fill(typemax(UInt64), chunkDims)
        else
            chunks = zeros(UInt64, chunkDims)
        end

        if value != false
            rest = 64 - (cols % 64)
            if rest < 64
                for i in 1:rows
                    chunks[i,end] = (chunks[i,end] << rest) >> rest
                end
            end
        end

        len = rows * cols
        bmat = new(chunks, len)
        bmat.dims = (rows, cols)

        return bmat
    end

    function BitTensor{3}(value, rows::Int, cols::Int, depth::Int)
        chunkDims = (rows, ceil(Int, cols / 64), depth)
        if value == undef
            chunks = rand(UInt64, chunkDims)
        elseif value == true
            chunks = fill(typemax(UInt64), chunkDims)
        else
            chunks = zeros(UInt64, chunkDims)
        end

        if value != false
            rest = 64 - cols % 64
            if rest < 64
                for k in 1:depth
                    for i in 1:rows
                        chunks[i,end,k] = (chunks[i,end,k] << rest) >> rest
                    end
                end
            end
        end

        len = rows * cols * depth
        bmat = new(chunks, len)
        bmat.dims = (rows, cols, depth)

        return bmat
    end

    function BitTensor{N}(value, dims::Vararg{Int, N}) where {N}
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

            if value != false
                rest = 64 - dims[2] % 64
                if rest < 64
                    counters = ones(Int, length(chunkDims) - 2)
                    limits = chunkDims[3:end]
                    llim = chunkDims[end]

                    while llim > counters[end]
                        for i in 1:chunkDims[1]
                            chunks[i,end,counters...] = (chunks[i,end,counters...] << rest) >> rest
                        end
                        for i in 1:(length(counters)-1)
                            if counters[i] == limits[i]
                                counters[i] = 1
                                counters[i+1] += 1
                            else
                                counters[i] += 1
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

@inline Base.length(mat::BitTensor) = mat.len

@inline function Base.getindex(mat::BitTensor{2}, i::Int, j::Int)
    jMod = (j - 1)  % 64
    return Bool((mat.chunks[i, ceil(Int, j / 64)] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BitTensor{3}, i::Int, j::Int, k::Int)
    jMod = (j - 1) % 64
    return Bool((mat.chunks[i, ceil(Int, j / 64), k] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BitTensor{4}, i::Int, j::Int, k::Int, l::Int)
    jMod = (j - 1) % 64
    return Bool((mat.chunks[i, ceil(Int, j / 64), k, l] & (1 << jMod)) >> jMod)
end
@inline function Base.getindex(mat::BitTensor, i::Int, j::Int, lastDims::Vararg{Int, N}) where {N}
    jMod = (j - 1) % 64
    return Bool((mat.chunks[i, ceil(Int, j / 64), lastDims...] & (1 << jMod)) >> jMod)
end

@inline setBit(mat::BitTensor, i::Int, j::Int, lastDims::Vararg{Int, N}) where {N} = 
    mat.chunks[i, ceil(Int, j / 64), lastDims...] |= 1 << ((j-1) % 64)

@inline clearBit(mat::BitTensor, i::Int, j::Int, lastDims::Vararg{Int, N}) where {N} =
    mat.chunks[i, ceil(Int, j / 64), lastDims...] &= ~(1 << ((j-1) % 64))

@inline function Base.setindex!(mat::BitTensor{2}, value::Bool, i::Int, j::Int)
    jMod = (j - 1) % 64
    mat.chunks[i, ceil(Int, j / 64)] &= ~(1 << jMod)
    mat.chunks[i, ceil(Int, j / 64)] |= value << jMod 
end
@inline function Base.setindex!(mat::BitTensor{3}, value::Bool, i::Int, j::Int, k::Int)
    jMod = (j - 1) % 64
    mat.chunks[i, ceil(Int, j / 64), k] &= ~(1 << jMod)
    mat.chunks[i, ceil(Int, j / 64), k] |= value << jMod 
end
@inline function Base.setindex!(mat::BitTensor{4}, value::Bool, i::Int, j::Int, k::Int, l::Int)
    jMod = (j - 1) % 64
    mat.chunks[i, ceil(Int, j / 64), k, l] &= ~(1 << jMod)
    mat.chunks[i, ceil(Int, j / 64), k, l] |= value << jMod 
end
@inline function Base.setindex!(mat::BitTensor, value::Bool, i::Int, j::Int, lastDims::Vararg{Int, N}) where{N}
    jMod = (j - 1) % 64
    mat.chunks[i, ceil(Int, j / 64), lastDims...] &= ~(1 << jMod)
    mat.chunks[i, ceil(Int, j / 64), lastDims...] |= value << jMod 
end