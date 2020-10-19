struct BitDense{S, T, O, F}
    #main data
    W::S
    b::T

    #float data
    isFloatOutput::O
    σ::F
end

function BitDense(in::Integer, out::Integer, 
            isFloatOutput::Bool = false, σ::Function = identity; 
            initW=undef, initB=undef)
        return BitDense(BitTensor{2}(initW, out, in), BitTensor{1}(initB, out), 
            isFloatOutput, σ)
end

@inline outputBitOperation(l::BitDense, temp::Int, b::Bool, wNumber::Int) = 
    (temp + b) > (!b + wNumber - temp)

@inline outputFloatOperation(l::BitDense, temp::Int, b::Bool, wNumber::Int) = 
    l.σ(1 - (temp + b) / wNumber)

function denseOperation(l::BitDense, x::BitArray, 
        out::Union{Array, BitArray}, operation::Function)
    Wc, b, Xc = l.W.chunks, l.b, x.chunks
    neuronNumber, weightsNumber, = size(Wc, 2), size(l.W, 2)
    wChunksNumber = ceil(Int, weightsNumber / 64.0)

    for i in 1:neuronNumber
        temp = 0
        for j in 1:wChunksNumber
            temp += count_ones(Wc[j,i] ⊻ Xc[j])
        end
        @inbounds out[i] = operation(l, temp, b[i], weightsNumber)
    end
    
    return out
end

@inline denseFloatExecute(l::BitDense, x::BitArray) = 
    return denseOperation(l, x,
        Array{Float32, 1}(undef, size(l.W,1)),
        outputFloatOperation)

@inline denseBitExecute(l::BitDense, x::BitArray) = 
    return denseOperation(l, x,
        BitArray{1}(undef, size(l.W,1)),
        outputBitOperation)

@inline function (l::BitDense)(x::BitArray)
    if l.isFloatOutput
        return denseFloatExecute(l, x)
    else
        return denseBitExecute(l, x)
    end
end

@inline (l::BitDense)(x::AbstractArray) = 
    l(signbit.(x))

#Flux Overloading
#=
struct TrainingData{N} <: AbstractArray{Float32, N}
    bit::Union{BitTensor{N}, BitArray{N}}
    float::Array{Float32, N}
end

TrainingBiases(bit::BitArray{1}) = TrainingData{1}(bit, Float32.(bit))
TrainingWeights(bit::BitTensor{2}) = TrainingData{2}(bit, Float32.(bit))

@inline Base.size(t::TrainingData) = size(t.float)
@inline Base.getindex(t::TrainingData, i::Int) = 
    t.float[i]
@inline Base.getindex(t::TrainingData, I::Vararg{Int, N}) where {N} =
    t.float[I...]
@inline Base.setindex!(t::TrainingData, value::Float32, i::Int) = 
    t.float[i] = value
@inline Base.setindex!(t::TrainingData, value::Float32, I::Vararg{Int, N}) where {N} = 
    t.float[i] = value
@inline getFloatArray(t::TrainingData) = t.float
@inline getBitArray(t::TrainingData) = t.bit



Flux.trainable(d::BitDense) = (TrainingWeights(d.W), 
                                TrainingBiases(d.b))

function Flux.update!(opt, x::TrainingData, x̄)
    Flux.update!(opt, getFloatArray(x), x̄)
    getBitArray(x) .= signbit.(getFloatArray(x))
end
function newDenseFromBitDense(l::BitDense)
    out = Dense(size(l.W, 2), size(l.W, 1))
    out.W .= l.W
    out.b .= l.b
    out
end

function newBitDenseFromDense(l::Dense)
    out = BitDense(size(l.W, 2), size(l.W, 1))
    out.W .= signbit.(l.W)
    out.b .= signbit.(l.b)
    out
end

function convertDenseToBitDense!(l::Dense, b::BitDense)
    b.W .= signbit.(l.W)
    b.b .= signbit.(l.b)
end

function newFloatChainFromBit(net::Chain)
    newLayers = []
    for layer in net.layers
        if layer isa BitDense
            push!(newLayers, newDenseFromBitDense(layer))
        else
            push!(newLayers, layer)
        end
    end
    Chain(newLayers...)
end

function convertChainToBit!(original::Chain, trained::Chain)
    for (ol, tl) in zip(original.layers, trained.layers)
        if ol isa BitDense
            convertDenseToBitDense!(tl, ol)
        end
    end
end

function train!(loss, chain::Chain, data, opt; cb = () -> ())
    fChain = newFloatChainFromBit(chain)
    Flux.train!(loss, params(fChain), data, opt, cb=cb)
    convertChainToBit!(chain, fChain)
end
=#

#Julia Base
function Base.show(io::IO, l::BitDense)
    if l.isFloatOutput
        print(io, "BitDense($(size(l.W,2)), $(size(l.W,1)), σ=$(l.σ))")
    else
        print(io, "BitDense($(size(l.W,2)), $(size(l.W,1)))")
    end
end

#=Base.show(io::IO, t::TrainingData) = 
    print(io, "TrainingData($(size(t.float,1))x$(size(t.float,2)))")=#
