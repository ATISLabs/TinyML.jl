function mutation_rand!(set::TrainingSet)
    cands = children(set)
    for child in cands
        layers = network(child).layers
        for i in indexes(set)
            mutate!(set, layers[i])
        end
    end
end

@inline function mutate!(set::TrainingSet, net::Union{Dense,BitDense})
    w, b, rate = weight(net), bias(net), mutation_rate(set)

    for i in 1:size(w,2)
        for j in 1:size(w,1)
            w[j,i] = randomizeweight(w, j, i, rate)
        end
    end
    for i in 1:length(b)
        b[i] = randomizebias(b, i, rate)
    end
end

#= Randoms =#
@inline rand_bias(W::Array{Float32, N}) where {N} = rand_bias()
@inline rand_bias(W::BitTensor) = rand(Bool)
@inline rand_bias(W::BitArray) = rand(Bool)

@inline rand_weight(W::Array{Float32, N}) where {N} = rand_weight()
@inline rand_weight(W::BitTensor) = rand(Bool)
@inline rand_weight(W::BitArray) = rand(Bool)

@inline randomizeweight(W::AbstractArray, i::Int, j::Int, rate::Float64) = 
                                    rand() < rate ? rand_weight(W) : W[i,j]
@inline randomizebias(bias::AbstractArray, i::Int, rate::Float64) = 
                                    rand() < rate ? rand_bias(bias) : bias[i]
