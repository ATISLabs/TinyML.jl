@inline randBias(W::Array{Float32, N}) where {N} = randBias()
@inline randBias(W::BitTensor) = rand(Bool)
@inline randBias(W::BitArray) = rand(Bool)

@inline randWeight(W::Array{Float32, N}) where {N} = randWeight()
@inline randWeight(W::BitTensor) = rand(Bool)
@inline randWeight(W::BitArray) = rand(Bool)

@inline randomizeWeight(W::AbstractArray, i::Int, j::Int, rate::Float64) = 
                                    rand() < rate ? randWeight(W) : W[i,j]
@inline randomizeBias(bias::AbstractArray, i::Int, rate::Float64) = 
                                    rand() < rate ? randBias(bias) : bias[i]

@inline function mutate!(set::TrainingSet, net::Union{Dense,BitDense})
    w, b, rate = net.W, net.b, getMutationRate(set)

    for i in 1:size(w,2)
        for j in 1:size(w,1)
            w[j,i] = randomizeWeight(w, j, i, rate)
        end
    end
    for i in 1:length(b)
        b[i] = randomizeBias(b, i, rate)
    end
end

function mutation!(set::TrainingSet)
    cands = getChildren(set)
    Threads.@threads for child in cands
        layers = getNetwork(child).layers
        for i in set.indexes
            mutate!(set, layers[i])
        end
    end
end