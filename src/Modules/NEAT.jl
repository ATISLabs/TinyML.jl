module NEAT
    #= Libraries =#
    using Distributions: Uniform
    using Flux

    #= Local imports =#
    include("./BitFlux.jl")
    using .BitFlux

    #= Includes =#
    include("./NEAT/Base.jl")
    include("./NEAT/Forward.jl")
    include("./NEAT/Evaluate.jl")
    include("./NEAT/Crossover.jl")
    include("./NEAT/Selection.jl")
    include("./NEAT/Mutation.jl")

    #= Exports =#
    export train!, NEATDense, TrainingSet, getChain

    #= Forward =#
    function (l::NEATDense)(input::Array{<:Number,1})
        forward(l, input, l.σ)
    end

    function (n::Network)(set::TrainingSet, input::Array{<:Number,1})
        forward(n, input, set.σ)
    end

    #= Genetic =#
    function train!(set::TrainingSet; genNumber::Int=typemax(Int64), 
            maxFitness::Float64 = Inf64)
        children::Array{Network} = []
        if set.isTrained
            children = crossover!(set)
            mutation!(set, children)
        else
            children = getFirstChildren(set)
            evaluate!(set, children)
            set.isTrained = true
        end

        gen = 0
        while gen < genNumber && 
                getFitness(unsafeGetRepresentant(set.species[1])) < maxFitness
            selection!(set, children)
            children = crossover!(set)
            mutation!(set, children)
            evaluate!(set, children)
            gen += 1
        end
        selection!(set, children)

        updateDense!(set)
        return set
    end
end