@inline getRandomWeight(W::Array{Float32, N}) where {N} = rand(Uniform(-1, 1))
@inline getRandomWeight(W::BitTensor) = rand(Bool)
@inline getRandomWeight(W::BitArray) = rand(Bool)

@inline randomizeWeight(W::AbstractArray, i::Int, j::Int, rate::Float64) = 
                                    rand() < rate ? getRandomWeight(W) : W[i,j]
@inline randomizeBias(bias::AbstractArray, i::Int, rate::Float64) = 
                                    rand() < rate ? getRandomWeight(bias) : bias[i]

@inline function mutate!(set::TrainingSet, net::Union{Dense,BitDense})
    w, b, rate = net.W, net.b, getMutationRate(set)

    for i in 1:size(w,1)
        for j in 1:size(w,2)
            w[i,j] = randomizeWeight(w, i, j, rate)
        end
        b[i] = randomizeBias(b, i, rate)
    end
end

"""
    genMutation(cands::Array{<:Candidate, 1}, selectedCandidates::Int, mutationRate::Float64)

Randomize some weights of the not selected candidates based on a probability of 'mutationRate'.

# Example
```julia_repl
julia> genMutation(candidates, 10, mutationRate)
```
In the above example, consider a candidates array of 100 subjects.
The first 10 subjects of the array are the ones we assume that were selected before.
The method executes for the 90 rest subject neurons weights a mutation with random values based on
on 'mutationRate' as a probability of mutation.
"""
function mutation!(set::TrainingSet)
    cands = view(getCandidates(set), getElitism(set)+1:length(getCandidates(set)))
    Threads.@threads for child in cands
        layers = getNetwork(child).layers
        for i in set.indexes
            mutate!(set, layers[i])
        end
    end
end