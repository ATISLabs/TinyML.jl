struct TrainingSet
    chain::Chain
    indexes::Tuple

    candidates::Array{Candidate, 1}

    population_size::Int
    elitism::Ref{Int}
    crossover_divisor::Ref{Int}
    mutation_rate::Ref{Float64}
    evals_per_candidate::Ref{Int}
end

function TrainingSet(chain::Chain, layers::Tuple; 
                        evals_per_candidate::Int=1,
                        population_size::Int=100,
                        elitism::Int=10,
                        crossover_divisor::Int=2,
                        mutation_rate::Float64=0.1)
    set = TrainingSet(chain,
                    generate_indexes(chain, layers),
                    Array{Candidate, 1}(undef, population_size),
                    population_size,
                    Ref(elitism),
                    Ref(crossover_divisor),
                    Ref(mutation_rate),
                    Ref(evals_per_candidate))
    initial_population!(set)
    set
end

function generate_indexes(chain::Chain, layers::Tuple)
    indexes = Array{Int, 1}(undef, length(layers))

    for (i, chainLayer) in enumerate(chain.layers)
        for (j, layer) in enumerate(layers)
            if chainLayer == layer
                indexes[j] = i
                break
            end
        end
    end
    Tuple(indexes...)
end

function initial_population!(set::TrainingSet)
    cands = candidates(set)
    for i in eachindex(candidates(set))
        cands[i] = Candidate(set)
    end
end

#= Candidate initialization =#
@inline copylayer(layer) = deepcopy(layer)

function Candidate(set::TrainingSet, copy_layers::Union{AbstractArray, Tuple}; 
        randomize::Bool=false)
    layers = collect(set.chain.layers)
    
    if randomize
        for i in indexes(set)
            layers[i] = randcopylayer(copy_layers[i])
        end
    else
        for i in indexes(set)
            layers[i] = copylayer(copy_layers[i])
        end
    end

    return Candidate(Chain(layers...))
end

@inline Candidate(set::TrainingSet) = 
    Candidate(set, set.chain.layers, randomize=true)
@inline Candidate(set::TrainingSet, cand::Candidate) = 
    Candidate(set, cand.net.layers, randomize=false)

#= TrainingSet Utils =#
sort!(set::TrainingSet) = 
    Base.sort!(candidates(set), by=x->fitness(x), rev=true)

@inline best(set::TrainingSet) = 
    view(candidates(set), 1:elitism(set))

@inline children(set::TrainingSet) = 
    view(candidates(set), (elitism(set)+1):population_size(set))
    
@inline Base.first(set::TrainingSet) = candidates(set)[1]

function best!(set::TrainingSet)
    sort!(set)
    return unsafe_best(set)
end

#= Getters and setters =#
@inline chain(set::TrainingSet) = set.chain
@inline population_size(set::TrainingSet) = set.population_size
@inline candidates(set::TrainingSet) = set.candidates

@inline evals_per_candidate(set::TrainingSet) = set.evals_per_candidate[]
@inline evals_per_candidate(set::TrainingSet, value::Int) = 
    set.evalsPerCandidate[] = value

@inline elitism(set::TrainingSet) = set.elitism[]
@inline elitism(set::TrainingSet, value::Int) = set.elitism[] = value

@inline crossover_divisor(set::TrainingSet) = set.crossover_divisor[]
@inline crossover_divisor(set::TrainingSet, value::Int) = 
    set.crossoverDivisor[] = value

@inline mutation_rate(set::TrainingSet) = set.mutation_rate[]
@inline mutation_rate(set::TrainingSet, value::Float64) = 
    set.mutationRate[] = value

@inline indexes(set::TrainingSet) = set.indexes

#= Displays =#
function Base.show(io::IO, t::TrainingSet)
    print(io, "TrainingSet(population_size=$(population_size(t)))")
end

function Base.display(t::TrainingSet)
    print("""Genetic TrainingSet:
    Settings:
    ├ Population size: $(population_size(t))
    ├ Elitism: $(elitism(t))
    ├ Crossover divisor: $(crossover_divisor(t))
    ├ Mutation rate: $(mutation_rate(t)),
    └ Evaluations per candidate: $(evals_per_candidate(t))
    Stats:
    └ Best fitness: $(bestfitness(t))
    """)
end