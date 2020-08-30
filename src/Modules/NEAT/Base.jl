#= Abstract types =#
abstract type NEATLayer end

#= Structs =#
mutable struct Connection
    in::Int
    out::Int

    weight::Float32
    enabled::Bool

    function Connection(in::Int, out::Int; outputState=false)
        c = new()

        c.in = in
        c.out = out
        c.weight = rand(Uniform(-1, 1))
        c.enabled = true

        return c
    end

    function Connection(con::Connection)
        c = new()

        c.in = con.in
        c.out = con.out
        c.weight = con.weight
        c.enabled = true

        return c
    end
end

mutable struct Node
    output::Float32
    bias::Float32

    connections::Array{Connection, 1}
    outputState::Bool

    function Node()
        n = new()

        n.connections = []
        n.outputState = false
        n.output = 0
        n.bias = rand(Uniform(-1,1))

        return n
    end
end

mutable struct NEATDense <: NEATLayer
    in::Int
    out::Int
    outputState::Bool

    nodes::Dict{Int, Node}
    outputNodes::Array{Node, 1}

    σ::Function

    function NEATDense(in::Int, out::Int, σ=identity)
        l = new()

        l.in = in
        l.out = out
        l.σ = σ
        initializeNetwork!(l, in, out)

        return l
    end
end

mutable struct Network <: NEATLayer
    nodes::Dict{Int, Node}
    outputNodes::Array{Node, 1}

    connections::Dict{Tuple{Int,Int}, Connection}
    innovations::Dict{Int, Connection}
    maxInnov::Int

    fitness::Float64

    outputState::Bool

    function Network(in::Int, out::Int)
        n = new()

        initializeNetwork!(n, in, out)
        n.connections = Dict{Tuple{Int,Int},Connection}()
        n.innovations = Dict{Int, Connection}()
        n.fitness = 0
        n.maxInnov = 0

        return n
    end

    Network(l::NEATDense) = Network(l.in, l.out)
end

mutable struct Specie
    candidates::Array{Network,1}

    function Specie()
        s = new()

        s.candidates = []

        return s
    end
end

mutable struct TrainingSet
    #Basic
    chain::Chain
    layer::NEATDense

    σ::Function
    in::Int
    out::Int
    popSize::Int
    maxSpecies::Int
    maxPopulation::Int
    isTrained::Bool

    #Evaluation
    fitnessFunc::Function

    #Selection
    species::Array{Specie, 1}
    survivalRate::Float64
    deltaThreshold::Float64
    c1::Float64
    c2::Float64
    c3::Float64

    #Crossover
    innovations::Dict{Tuple{Int,Int}, Int}
    innovationNumber::Int
    reproductionRate::Float64

    #Mutation
    biasMutationRate::Float64
    weightMutationRate::Float64
    toggleConnectionMutationRate::Float64
    addNodeMutationRate::Float64
    addConnectionMutationRate::Float64

    function TrainingSet(chain::Chain,
            layer::NEATDense,
            fitnessFunc::Function;
            c1=0.5,
            c2=0.5,
            c3=0.5,
            maxPopulation=400,
            maxSpecies=10,
            survivalRate=0.4,
            deltaThreshold=0.7, 
            reproductionRate=1,
            biasMutationRate=0.1,
            weightMutationRate=0.2,
            toggleConnectionMutationRate=0.1,
            addNodeMutationRate=0.1,
            addConnectionMutationRate=0.1)
        n = new()

        n.chain = chain
        n.layer = layer

        n.isTrained = false
        n.layer = layer
        n.σ = layer.σ
        n.in = layer.in
        n.out = layer.out
        n.fitnessFunc = fitnessFunc

        n.popSize = 0
        n.innovationNumber = 1
        n.innovations = Dict{Tuple{Int,Int}, Int}()

        n.c1 = c1
        n.c2 = c2
        n.c3 = c3
        n.maxSpecies = maxSpecies
        n.maxPopulation = maxPopulation
        n.survivalRate = survivalRate
        n.deltaThreshold = deltaThreshold
        n.reproductionRate = reproductionRate
        n.biasMutationRate = biasMutationRate
        n.weightMutationRate = weightMutationRate
        n.toggleConnectionMutationRate = toggleConnectionMutationRate
        n.addNodeMutationRate = addNodeMutationRate
        n.addConnectionMutationRate = addConnectionMutationRate

        n.species = [Specie()]
        push!(n.species[1], Network(layer))
        addRandomConnection!(n, n.species[1].candidates[1])

        return n            
    end
end

mutable struct EvaluationLayer
    net::Network
    σ::Function

    function EvaluationLayer(set::TrainingSet)
        e = new()

        e.σ = set.layer.σ

        return e
    end
end

#= Common/Other =#
#@inline randFloatWeight() = rand(Uniform(-1, 1))
#@inline randBinaryWeight() = rand([-1, 1])

@inline getChain(set::TrainingSet) = set.chain

function getEvaluationChain(set::TrainingSet)
    layers = []
    index = -1

    for (i, layer) in enumerate(set.chain.layers)
        if layer == set.layer
            push!(layers, EvaluationLayer(set))
            index = i
        else
            push!(layers, layer)
        end
    end

    return Chain(layers...), index
end

@inline unsafeReplaceNetwork!(ev::Chain, index::Int, n::Network) = 
        ev.layers[index].net = n

function replaceNetwork!(ev::Chain, n::Network)
    for layer in ev.layers
        if layer isa EvaluationLayer
            layer.net = n
        end
    end
end

function addRandomConnection!(set::TrainingSet, n::Network)
    #gets a random node from input nodes and hidden nodes as the connection input
    in = rand(1:(set.in + length(n.nodes) - set.out))
    #gets a random node from hidden nodes and output nodes as the connections output
    out = rand(1:length(n.nodes)) + set.in
    addConnection!(set, n, in, out)
end

function getFirstChildren(set::TrainingSet)
    n = Network(set.layer)
    addRandomConnection!(set, n)
    return [n]
end

function updateDense!(set::TrainingSet)
    topfit = -1
    top = nothing
    for specie in set.species
        rep = getRepresentant!(specie)
        if rep.fitness > topfit
            top = rep
            topfit = rep.fitness
        end
    end

    neat = set.layer
    neat.nodes = deepcopy(top.nodes)
    neat.outputNodes = []
    for i in (set.in+1):(set.out + set.in)
        push!(neat.outputNodes, neat.nodes[i])
    end
end

@inline addNode!(l::NEATDense) = unsafeAddNode!(l, l.in)

@inline addNode!(set::TrainingSet, l::Network) = unsafeAddNode!(l, set.in)

function unsafeAddNode!(l::NEATLayer, in::Int)
    id = in + length(l.nodes) + 1
    push!(l.nodes, id => Node())
    return id
end

@inline function initializeNetwork!(l::NEATLayer, in::Int, out::Int)
    l.outputState = false
    l.nodes = Dict{Int, Node}()
    l.outputNodes = []
    for i in 1:out
        push!(l.outputNodes, l.nodes[unsafeAddNode!(l, in)])
    end
end

@inline function addConnection!(set::TrainingSet, 
                                l::Network, in::Int, out::Int; 
                                con::Connection=Connection(in, out))
    #If it already has the key, mutate its weight overwriting the object
    if haskey(l.connections, (in, out))
        push!(l.connections, (in, out) => con)
    #If it doesn't have the key, create a new connection
    else
        innov = getInnovation!(set, in, out)
        if l.maxInnov < innov
            l.maxInnov = innov
        end
        push!(l.connections, (in, out) => con)
        push!(l.innovations, innov => con)
        push!(l.nodes[out].connections, con)
    end
end

@inline function addConnection!(l::NEATDense, in::Int, out::Int; 
        con::Connection=Connection(in,out))
    push!(l.nodes[out].connections,con)
end

@inline Base.length(s::Specie) = length(s.candidates)

@inline Base.getindex(s::Specie, i::Int) = s.candidates[i]

@inline Base.push!(s::Specie, cand::Network) = push!(s.candidates, cand)

@inline Base.sort!(s::Specie) = sort!(s.candidates, by=c->c.fitness, rev=true)

@inline getCandidates(s::Specie) = s.candidates

function getRepresentant!(s::Specie)
    sort!(s)
    return s.candidates[1]
end

function unsafeGetRepresentant(s::Specie)
    return s.candidates[1]
end

#= Displays =#
function Base.display(set::TrainingSet)
    println(`Number of species: $(length(set.species))`)
end

function Base.display(con::Connection)
    println(`Weight: $(con.weight) In: $(con.in) Out: $(con.out)`)
end

function Base.display(n::Node)
    println(`Bias: $(n.bias)`)
    display.(n.connections)
end

function Base.display(ca::NEATLayer)
    display(ca.nodes)
end

function Base.display(n::Network)
    println(`Fitness: $(n.fitness)`)
    display(n.nodes)
end

function Base.display(s::Specie)
    display.(s.candidates)
end