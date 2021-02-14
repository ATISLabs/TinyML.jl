#= Abstract types =#
abstract type NEATLayer end

#= Structs =#
mutable struct Connection
    in::Int
    out::Int

    weight::Float32
    enabled::Bool

    function Connection(in::Int, out::Int; outputState=false,
            w = randWeight(),
            enabled = true)
        c = new()

        c.in = in
        c.out = out
        c.weight = w
        c.enabled = enabled

        c
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
    id::Int

    output::Float32
    bias::Float32

    connections::Array{Connection, 1}
    outputState::Bool

    function Node(id::Int)
        n = new()

        n.id = id
        n.connections = []
        n.outputState = false
        n.output = 0
        n.bias = randBias()

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
        initializeCandidate!(l, in, out)

        return l
    end
end

mutable struct Candidate <: NEATLayer
    nodes::Dict{Int, Node}
    outputNodes::Array{Node, 1}

    connections::Dict{Tuple{Int,Int}, Connection}
    innovations::Dict{Int, Connection}
    maxInnov::Int

    layers::Array{Array{Node, 1}, 1}

    fitness::Float64

    outputState::Bool

    function Candidate(in::Int, out::Int)
        n = new()

        initializeCandidate!(n, in, out)
        n.layers = [n.outputNodes]
        n.connections = Dict{Tuple{Int,Int},Connection}()
        n.innovations = Dict{Int, Connection}()
        n.fitness = 0
        n.maxInnov = 0

        return n
    end

    Candidate(l::NEATDense) = Candidate(l.in, l.out)
end

mutable struct Specie
    candidates::Array{Candidate,1}

    function Specie()
        s = new()

        s.candidates = []

        return s
    end
end

mutable struct TrainingSet{netType}
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
    evalsPerCandidate::Int
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
            feedForward::Bool=true,
            evalsPerCandidate::Int=1,
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
        n = new{feedForward ? :DFF : :NEAT}()

        n.chain = chain
        n.layer = layer

        n.isTrained = false
        n.layer = layer
        n.evalsPerCandidate = evalsPerCandidate
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
        push!(n.species[1], Candidate(layer))
        addRandomConnection!(n, n.species[1].candidates[1])

        return n            
    end
end

mutable struct EvaluationLayer
    net::Candidate
    σ::Function

    function EvaluationLayer(set::TrainingSet)
        e = new()

        e.σ = set.layer.σ

        return e
    end
end

#= Common/Other =#
@inline randWeight() = Float32(rand(Uniform(-1, 1)))
@inline randBias() = Float32(rand(Uniform(-1,1)))

@inline getNode(cand::Candidate, id::Int) = cand.nodes[id]
@inline getId(node::Node) = node.id
@inline getLayers(cand::Candidate) = cand.layers
@inline getIn(set::TrainingSet) = set.in
@inline getEvalsPerCandidate(set::TrainingSet) = set.evalsPerCandidate
@inline getChain(set::TrainingSet) = set.chain
@inline isTrained(set::TrainingSet) = set.isTrained
@inline setTrained!(set::TrainingSet) = set.isTrained = true
function isTrained!(set::TrainingSet)
    setTrained!(set::TrainingSet)
    return false
end
@inline getLayer(set::TrainingSet) = set.layer
@inline getFitness(candidate::Candidate) = candidate.fitness

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

@inline unsafeReplaceCandidate!(ev::Chain, index::Int, n::Candidate) = 
        ev.layers[index].net = n

function replaceCandidate!(ev::Chain, n::Candidate)
    for layer in ev.layers
        if layer isa EvaluationLayer
            layer.net = n
        end
    end
end

function addRandomConnection!(set::TrainingSet, n::Candidate)
    #gets a random node from input nodes or hidden nodes as the connection input
    in = rand(1:(set.in + length(n.nodes) - set.out))
    #gets a random node from hidden nodes or output nodes as the connection output
    out = rand(1:length(n.nodes)) + set.in
    addConnection!(set, n, in, out)
end

function addRandomConnection!(set::TrainingSet{:DFF}, n::Candidate)
    index = rand(1:length(n.layers))
    if index == 1
        in = rand(1:getIn(set))
    else
        in = getId(rand(n.layers[index-1]))
    end
    out = getId(rand(n.layers[index]))
    addConnection!(set, n, in, out)
end

function getFirstChildren(set::TrainingSet)
    n = Candidate(set.layer)
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
    neat.outputNodes = [neat.nodes[getId(node)] for node in top.outputNodes]
end

@inline addNode!(l::NEATDense) = unsafeAddNode!(l, l.in)

@inline addNode!(set::TrainingSet, l::Candidate) = unsafeAddNode!(l, set.in)

function unsafeAddNode!(l::NEATLayer, in::Int)
    id = in + length(l.nodes) + 1
    push!(l.nodes, id => Node(id))
    return id
end

@inline function initializeCandidate!(l::NEATLayer, in::Int, out::Int)
    l.outputState = false
    l.nodes = Dict{Int, Node}()
    l.outputNodes = []
    for i in 1:out
        push!(l.outputNodes, l.nodes[unsafeAddNode!(l, in)])
    end
end

@inline function addConnection!(set::TrainingSet, 
                                l::Candidate, 
                                in::Int,
                                out::Int;
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

@inline Base.push!(s::Specie, cand::Candidate) = push!(s.candidates, cand)

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
function Base.show(io::IO, con::Connection)
    print(io, "Connection(w=$(con.weight), i=$(con.in), o=$(con.out))")
end

function Base.show(io::IO, n::Node)
    print(io, "Node(b=$(n.bias) conns=$(length(n.connections)))")
end

function Base.show(io::IO, ca::NEATLayer)
    print(io, "NEATLayer")
end

function Base.show(io::IO, n::Candidate)
    print(io, "Candidate(fit=$(n.fitness), nodes=$(length(n.nodes)), conns=$(length(n.connections)))")
end

function Base.show(io::IO, n::NEATDense)
    print(io, "NEATDense(i=$(n.in), o=$(n.out), nodes=$(length(n.nodes)))")
end

function Base.show(io::IO, s::Specie)
    print(io, "Specie($(length(s.candidates)))")
end

function Base.display(s::Specie)
    display(s.candidates)
end

function Base.show(io::IO, l::TrainingSet)
    print(io, "NEAT_TrainingSet(pop=$(l.popSize))")
end

function Base.display(l::TrainingSet)
    print("""NEAT_TrainingSet
Basic
    Max population: $(l.maxPopulation)
    Max species: $(l.maxSpecies)
Evaluation
    Evaluation per candidate: $(l.evalsPerCandidate)
    Fitness function: $(l.fitnessFunc)
Selection
    Survival rate: $(l.survivalRate)
    Delta threshold: $(l.deltaThreshold)
    c1: $(l.c1)
    c2: $(l.c2)
    c3: $(l.c3)
Crossover
    Last innovation number: $(l.innovationNumber)
    Reproduction rate: $(l.reproductionRate)
Mutation
    Bias mutation rate: $(l.biasMutationRate)
    Weight mutation rate: $(l.weightMutationRate)
    Toggle connection mutation rate: $(l.toggleConnectionMutationRate)
    Add node mutation rate: $(l.addNodeMutationRate)
    Add connection mutation rate: $(l.addConnectionMutationRate)
Variable
    Population size: $(l.popSize)
    Number of species: $(length(l.species))
    """)
end