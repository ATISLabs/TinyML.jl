struct Candidate
    net::Chain
    fitness::Ref{Float32}
end

@inline Candidate(net::Chain) = Candidate(net, Ref(0.f0))

setFitness!(cand::Candidate, fitness::AbstractFloat) = cand.fitness[] = Float32(fitness)
getFitness(cand::Candidate) = cand.fitness[]
getNetwork(cand::Candidate) = cand.net


"""
    mutable struct TrainingSet

Stores all data for genetic algorithm.

# Structure
```julia
candidates::Array{<:Candidate, 1}
fitnessFunc::Function

popSize::Int64
selectedCandidatesCount::Int64
crossoverCuttingFractionDivisor::Int64
mutationRate::Float64
```
- candidates: Stores an array of candidates to be tested
- fitnessFunc: Stores the function that evaluates candidates
- popSize: Number of candidates
- selectedCandidatesCount: Number of candidates to be selected to the next generation
- crossoverCuttingFractionDivisor: Number used to find crossover point by splitting neurons
     weights on crossover (crossoverPoint = numberOfWeightsPerNeuron / crossoverCuttingFractionDivisor)
- mutationRate: Probability of mutation for each weight

# Initialization
```julia
TrainingSet(candidates::Array{<:Candidate, 1}, fitnessFunc::Function, populationSize::Int64,
            selectedCandidatesNumber::Int64, crossoverDivisor::Int64, mutationRate::Float64)
```

# Example
```julia_repl
julia> TrainingSet(snakeCandidates, snakeFitness, 2000, 10, 2, 0.1)
```
"""
struct TrainingSet{N}
    isTrained::Ref{Bool}

    chain::Chain
    indexes::NTuple{N, Int}

    candidates::Array{Candidate, 1}
    fitnessFunc::Function

    popSize::Int
    elitism::Ref{Int}
    crossoverDivisor::Ref{Int}
    mutationRate::Ref{Float64}
end

function setParameter!(set::TrainingSet;
                    isTrained::Bool=set.isTrained,
                    elitism::Int=set.elitism,
                    crossoverDivisor::Int=set.crossoverDivisor,
                    mutationRate::Float64=set.mutationRate)
    set.isTrained[] = isTrained
    set.elitism[] = elitism
    set.crossoverDivisor[] = crossoverDivisor
    set.mutationRate[] = mutationRate
end

@inline getCandidates(set::TrainingSet) = set.candidates
@inline isTrained(set::TrainingSet) = set.isTrained[]
@inline getElitism(set::TrainingSet) = set.elitism[]
@inline getCrossoverDivisor(set::TrainingSet) = set.crossoverDivisor[]
@inline getMutationRate(set::TrainingSet) = set.mutationRate[]

function getIndexes(chain::Chain, layers::Tuple)
    indexes = Array{Int, 1}(undef, length(layers))

    for (i, chainLayer) in enumerate(chain.layers)
        for (j, layer) in enumerate(layers)
            if chainLayer == layer
                indexes[j] = i
                break
            end
        end
    end

    return Tuple(indexes)
end

function generateInitialPopulation!(set::TrainingSet)
    for i in 1:length(set.candidates)
        set.candidates[i] = Candidate(set)
    end
end

function TrainingSet(chain::Chain, layers::Tuple, fitnessFunc::Function; 
                                popSize::Int=100,
                                elitism::Int=10,
                                crossoverDivisor::Int=2,
                                mutationRate::Float64=0.1)
    set = TrainingSet(Ref(false),
                    chain,
                    getIndexes(chain, layers),
                    Array{Candidate, 1}(undef, popSize),
                    fitnessFunc,
                    popSize,
                    Ref(elitism),
                    Ref(crossoverDivisor),
                    Ref(mutationRate))
    generateInitialPopulation!(set)

    return set
end

@inline getRandomizedLayerCopy(dense::Dense) = 
    Dense(size(dense.W, 2), size(dense.W, 1), dense.σ)
@inline getRandomizedLayerCopy(dense::BitDense) = 
    BitDense(size(dense.W, 2), size(dense.W, 1), dense.isFloatOutput, dense.σ)

#= Candidate Initialization =#
function Candidate(set::TrainingSet, copyLayers; randomize::Bool=false)
    layers = collect(set.chain.layers)
    
    if randomize
        for i in set.indexes
            layers[i] = getRandomizedLayerCopy(copyLayers[i])
        end
    else
        for i in set.indexes
            layers[i] = deepcopy(copyLayers[i])
        end
    end

    return Candidate(Chain(layers...))
end

@inline Candidate(set::TrainingSet) = Candidate(set, set.chain.layers, randomize=true)
@inline Candidate(set::TrainingSet, cand::Candidate) = Candidate(set, cand.net.layers, randomize=false)

function updateLayer!(old::Union{Dense, BitDense}, new::Union{Dense, BitDense})
    old.W .= new.W
    old.b .= new.b
end

function updateChain!(set::TrainingSet)
    sort!(set.candidates, by=x->getFitness(x), rev=true)

    for i in set.indexes
        updateLayer!(set.chain.layers[i], set.candidates[1].net.layers[i])
    end
end

#= Common/Other =#