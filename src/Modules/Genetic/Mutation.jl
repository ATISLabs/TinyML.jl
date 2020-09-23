@inline getRandomWeight(W::Array{Float32, N}) where {N} = rand(Uniform(-1, 1))
@inline getRandomWeight(W::BitTensor) = rand(Bool)
@inline getRandomWeight(W::BitArray) = rand(Bool)

@inline randomizeWeight(W::AbstractArray, i::Int, j::Int, rate::Float64) = 
                                    rand() < rate ? getRandomWeight(W) : W[i,j]
@inline randomizeBias(bias::AbstractArray, i::Int, rate::Float64) = 
                                    rand() < rate ? getRandomWeight(bias) : bias[i]

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
    cands = view(getCandidates(set), getElitism(set)+1:length(getCandidates(set)))
    Threads.@threads for child in cands
        layers = getNetwork(child).layers
        for i in set.indexes
            mutate!(set, layers[i])
        end
    end
end