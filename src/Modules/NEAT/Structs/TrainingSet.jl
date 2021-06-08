struct TrainingSet{netType}
    #Basic
    chain::Chain
    layer::NEATDense

    σ::Function
    in::Int
    out::Int
    population_size::Ref{Int}
    max_species::Int
    max_population::Int

    #Evaluation
    evals_per_candidate::Int
    additional_data::Any

    #Selection
    species::Array{Specie, 1}
    survival_rate::Float64
    delta_threshold::Float64
    c1::Float64
    c2::Float64
    c3::Float64

    #Crossover
    innovations::Dict{Tuple{Int,Int}, Int}
    max_innovation_number::Ref{Int}
    reproduction_rate::Float64

    #Mutation
    bias_mutation_rate::Float64
    weight_mutation_rate::Float64
    toggle_connection_mutation_rate::Float64
    add_node_mutation_rate::Float64
    add_connection_mutation_rate::Float64
end

function TrainingSet(chain::Chain,
        layer::NEATDense;
        feed_forward::Bool=true,
        evals_per_candidate::Int=1,
        additional_data=nothing,
        c1::Float64=0.5,
        c2::Float64=0.5,
        c3::Float64=0.5,
        max_population::Int=400,
        max_species::Int=typemax(Int64),
        survival_rate::Float64=0.4,
        delta_threshold::Float64=0.7, 
        reproduction_rate::Float64=1,
        bias_mutation_rate::Float64=0.1,
        weight_mutation_rate::Float64=0.2,
        toggle_connection_mutation_rate::Float64=0.1,
        add_node_mutation_rate::Float64=0.1,
        add_connection_mutation_rate::Float64=0.1,
        σ::Function=identity)
    set = TrainingSet{feed_forward ? :DFF : :NEAT}(
        chain,
        layer,
        σ(layer),
        in(layer),
        out(layer),
        0,
        max_species,
        max_population,
        evals_per_candidate,
        additional_data,
        [Specie()],
        survival_rate,
        delta_threshold,
        c1,
        c2,
        c3,
        Dict{Tuple{Int, Int}, Connection}(),
        Ref(1),
        reproduction_rate,
        bias_mutation_rate,
        weight_mutation_rate,
        toggle_connection_mutation_rate
    )
    push!(species(set, 1), Candidate(layer))
    rand_connection!(set, species(set, 1), candidate(species(set, 1), 1))

    return n            
end

#Getters and Setters
@inline additional_data(set::TrainingSet) = set.additionalData
@inline in(set::TrainingSet) = set.in
@inline evals_per_candidate(set::TrainingSet) = set.evalsPerCandidate
@inline chain(set::TrainingSet) = set.chain
@inline layer(set::TrainingSet) = set.layer

@inline max_innovation_number(set::TrainingSet) = set.max_innovation_number[]
@inline function max_innovation_number(set::TrainingSet, val::Int)
    if (max_innovation_number(set) < val)
        set.max_innovation_number[] = val
    end
    max_innovation_number(set)
end