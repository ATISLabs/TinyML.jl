struct BitDense{S, T, O, F}
    W::S
    b::T
    isFloatOutput::O
    σ::F
end

function BitDense(in::Integer, out::Integer, isFloatOutput::Bool = false, σ::Function = identity;
        initW=undef, initB=undef)
        return BitDense(BitTensor{2}(initW, out, in), BitTensor{1}(initB, out), isFloatOutput, σ)
end

function bitDenseBitExecute(l::BitDense, x::BitArray{1})
    out = BitArray{1}(undef, size(l.W, 1))

    ccall((:binaryBitOutDotProduct, binarynetlib), Cvoid, 
        (Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}, Cint, Cint), 
        x.chunks, l.W.chunks, l.b.chunks, out.chunks, size(l.W, 1), size(l.W, 2))

    return out
end

#Old
function bitExecute(l::BitDense, x::BitArray{1})
    W, b = l.W.chunks, l.b

    ilim, jlim, wcnt = size(W, 1), size(W, 2), length(x)
    out = BitArray{1}(undef, ilim)

    Wx = x.chunks

    for i in 1:ilim
        temp = 0
        for j in 1:jlim
            temp += count_ones(W[i, j] ⊻ Wx[j])
        end
        @inbounds out[i] = (temp + b[i]) > (!b[i] + wcnt - temp)
    end
    
    return out
end

function bitDenseFloatExecute(l::BitDense, x::BitArray{1})
    out = Array{Float32, 1}(undef, size(l.W, 1))

    ccall((:binaryFloatOutDotProduct, binarynetlib), Cvoid, 
        (Ptr{UInt64}, Ptr{UInt64}, Ptr{UInt64}, Ptr{Float32}, Cint, Cint), 
        x.chunks, l.W.chunks, l.b.chunks, out, size(l.W, 1), size(l.W, 2))

    return l.σ.(out)
end

@inline function (l::BitDense)(x::BitArray)
    if l.isFloatOutput
        return bitDenseFloatExecute(l, x)
    else
        return bitDenseBitExecute(l, x)
    end
end

@inline (l::BitDense)(x::Array{<:Number, 1}) = l(BitArray([xi>0 for xi in x]))