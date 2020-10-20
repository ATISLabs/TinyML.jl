struct Candidate
    net::Chain
    fitness::Ref{Float64}
end

@inline Candidate(net::Chain) = Candidate(net, Ref(0.0))

setFitness!(cand::Candidate, fitness::AbstractFloat) = cand.fitness[] = Float64(fitness)
getFitness(cand::Candidate) = cand.fitness[]
getNetwork(cand::Candidate) = cand.net



struct TrainingSet{N}
    chain::Chain
    indexes::NTuple{N, Int}

    candidates::Array{Candidate, 1}
    fitnessFunc::Function

    trained::Ref{Bool}
    popSize::Int
    elitism::Ref{Int}
    crossoverDivisor::Ref{Int}
    mutationRate::Ref{Float64}
    evalsPerCandidate::Ref{Int}
end

function setParameter!(set::TrainingSet;
                    evalsPerCandidate::Int=set.evalsPerCandidate[],
                    elitism::Int=set.elitism[],
                    crossoverDivisor::Int=set.crossoverDivisor[],
                    mutationRate::Float64=set.mutationRate[])
    set.evalsPerCandidate[] = evalsPerCandidate
    set.elitism[] = elitism
    set.crossoverDivisor[] = crossoverDivisor
    set.mutationRate[] = mutationRate
end

@inline setTrained(set::TrainingSet) = set.trained[] = true
@inline isTrained(set::TrainingSet) = set.trained[]
function isTrained!(set::TrainingSet)
    setTrained(set)
    return false
end
@inline getPopSize(set::TrainingSet) = set.popSize
@inline getEvalsPerCandidate(set::TrainingSet) = set.evalsPerCandidate[]
@inline getCandidates(set::TrainingSet) = set.candidates
@inline getElitism(set::TrainingSet) = set.elitism[]
@inline getCrossoverDivisor(set::TrainingSet) = set.crossoverDivisor[]
@inline getMutationRate(set::TrainingSet) = set.mutationRate[]
@inline getFitnessFunction(set::TrainingSet) = set.fitnessFunc
@inline getBestPerformed(set::TrainingSet) = view(getCandidates(set), 
    1:getElitism(set))
@inline getChildren(set::TrainingSet) = view(getCandidates(set), 
    (getElitism(set)+1):getPopSize(set))

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
                                evalsPerCandidate::Int=1,
                                popSize::Int=100,
                                elitism::Int=10,
                                crossoverDivisor::Int=2,
                                mutationRate::Float64=0.1)
    set = TrainingSet(chain,
                    getIndexes(chain, layers),
                    Array{Candidate, 1}(undef, popSize),
                    fitnessFunc,
                    Ref(false),
                    popSize,
                    Ref(elitism),
                    Ref(crossoverDivisor),
                    Ref(mutationRate),
                    Ref(evalsPerCandidate))
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

sort!(set::TrainingSet) = 
    Base.sort!(getCandidates(set), by=x->getFitness(x), rev=true)

@inline unsafeGetBest(set::TrainingSet) = getCandidates(set)[1]

function getBest!(set::TrainingSet)
    sort!(set)
    return unsafeGetBest(set)
end

function updateChain!(set::TrainingSet)
    sort!(set)

    for i in set.indexes
        updateLayer!(getChain(set).layers[i], getNetwork(unsafeGetBest(set)).layers[i])
    end
end

#= Common/Other =#
@inline randWeight() = Float32(rand(Uniform(-1,1)))
@inline randBias() = Float32(rand(Uniform(-1,1)))

@inline getChain(t::TrainingSet) = t.chain

#= Displays =#
function Base.show(io::IO, t::Candidate)
    print(io, "Candidate(fit=$(getFitness(t))")
end

function Base.show(io::IO, t::TrainingSet)
    print(io, "TrainingSet(popSize=$(t.popSize))")
end

function Base.display(t::TrainingSet)
    print("""Genetic_TrainingSet
    Fitness function: $(t.fitnessFunc)
    Population size: $(t.popSize)
    Elitism: $(t.elitism)
    Crossover divisor: $(t.crossoverDivisor)
    Mutation rate: $(t.mutationRate),
    Evaluations per candidate: $(t.evalsPerCandidate)
    """)
end