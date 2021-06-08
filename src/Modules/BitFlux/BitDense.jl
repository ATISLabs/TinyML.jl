struct BitDense{output}
    #main data
    weight::BitTensor
    bias::BitArray

    #float data
    σ::Function
end

function BitDense(in::Integer, out::Integer, 
            isfloat::Bool = false, σ::Function = identity; 
            initW=undef, initB=undef)
    return BitDense{isfloat ? :Float : :Bit}(BitTensor{2}(initW, out, in), 
                                            BitTensor{1}(initB, out), σ)
end

@inline BitDense{:Float}(in::Int, out::Int, σ::Function, initW=undef, initB=undef) =
    BitDense(in, out, true, σ, initW = initW, initB = initB)
@inline BitDense{:Bit}(in::Int, out::Int, σ::Function, initW=undef, initB=undef) =
    BitDense(in, out, false, σ, initW = initW, initB = initB)

@inline operation(dense::BitDense{:Bit}, temp::Int, b::Bool, wnumber::Int) = 
    (temp + b) > (!b + wnumber - temp)

@inline operation(dense::BitDense{:Float}, temp::Int, b::Bool, wnumber::Int) = 
    σ(dense)(1 - (temp + b) / wnumber)

@inline alloc_out(dense::BitDense{:Float}) = zeros(Float32, outputnumber(dense))
@inline alloc_out(dense::BitDense{:Bit}) = falses(outputnumber(dense))

@inline function (dense::BitDense)(x::BitArray)
    out = alloc_out(dense)
    Wc, b_arr, Xc = chunks(weight(dense)), bias(dense), chunks(x)
    neuron_number, weights_number, = outputnumber(dense), inputnumber(dense)
    chunks_number = ceil(Int, weights_number / 64.0)

    for i in 1:neuron_number
        temp = 0
        for j in 1:chunks_number
            temp += count_ones(Wc[j,i] ⊻ Xc[j])
        end
        @inbounds out[i] = operation(dense, temp, b_arr[i], weights_number)
    end
    
    return out
end

@inline (dense::BitDense)(x::AbstractArray) = 
    dense(signbit.(x))

#Julia Base
function Base.show(io::IO, l::BitDense{:Float})
    print(io, "BitDense{:Float}($(size(weight(l),2)), $(size(bias(l),1)), σ=$(σ(l))")
end

function Base.show(io::IO, l::BitDense{:Bit})
    print(io, "BitDense{:Bit}($(size(weight(l),2)), $(size(bias(l),1)))")
end

#Auxiliar
@inline outputnumber(dense::BitDense) = size(weight(dense), 1)
@inline inputnumber(dense::BitDense) = size(weight(dense), 2)

#Getters and setters
@inline weight(dense::BitDense) = dense.weight
@inline bias(dense::BitDense) = dense.bias
@inline σ(dense::BitDense) = dense.σ