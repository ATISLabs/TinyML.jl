struct Candidate{T} <: AbstractNEATLayer
    nodes::Dict{Int, Node}

    input_nodes::Array{Node, 1}
    output_nodes::Array{Node, 1}

    connections::Dict{Tuple{Int,Int}, Connection}
    innovations::Dict{Int, Connection}
    innovation_number::Ref{Int}

    layers::Union{Array{Array{Node, 1}, 1}, Missing}

    fitness::Ref{Float32}

    state::Ref{Bool}
end

function Candidate(in::Int, out::Int, type::Symbol)
    candidate = Candidate{type}(
        Dict{Int, Node}(),
        [], [],
        Dict{Tuple{Int, Int}, Connection}(),
        Dict{Int, Connection}(),
        Ref(0),
        type == DFF_NETWORK_SYMBOL ? [] : missing,
        Ref(0.0f0),
        Ref(false)
    )
    initialize!(candidate, in(set), out(set))
    candidate
end

Candidate(l::NEATDense) = Candidate(l.in, l.out)

function push!(candidate::Candidate, node::InputNode)
    push!(input_nodes(candidate), node)
    push!(candidate, node)
end

#Getters and Setters
@inline nodes(candidate::Candidate) = candidate.nodes
@inline Base.length(candidate::Candidate) = length(nodes(candidate))

@inline input_nodes(candidate::Candidate) = candidate.input_nodes
@inline in(candidate::Candidate) = length(input_nodes(candidate))

@inline output_nodes(candidate::Candidate) = candidate.output_nodes
@inline out(candidate::Candidate) = length(output_nodes(candidate))

@inline connections(candidate::Candidate) = candidate.connections
@inline innovations(candidate::Candidate) = candidate.innovations
@inline layers(candidate::Candidate) = candidate.layers

@inline state(candidate::Candidate) = candidate.state[]
@inline state!(candidate::Candidate) = candidate.state[] != state(candidate)

@inline innovation_number(candidate::Candidate) = candidate.innovation_number[]
@inline innovation_number!(candidate::Candidate) = 
    candidate.innovation_number[] = innovation_number(candidate) + 1

@inline fitness(candidate::Candidate) = candidate.fitness[]
@inline fitness!(candidate::Candidate, fitness::Float32) = 
    candidate.fitness[] = fitness

@inline node(candidate::Candidate, id::Int) = candidate.nodes[id]
