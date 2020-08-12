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
    export Train!, NEATDense, TrainingSet, getChain

    #= Forward =#
    function (l::NEATDense)(input::Array{<:Number,1})
        forward(l, input, l.σ)
    end

    function (n::Network)(set::TrainingSet, input::Array{<:Number,1})
        forward(n, input, set.σ)
    end

    #= Genetic =#
    function Train!(set::TrainingSet, genCount::Int)
        children::Array{Network} = []
        if set.isTrained
            children = crossover!(set)
            mutation!(set, children)
        else
            children = getFirstChildren(set)
            evaluate!(set, children)
            set.isTrained = true
        end

        for gen in 1:genCount
            selection!(set, children)
            children = crossover!(set)
            mutation!(set, children)
            evaluate!(set, children)
        end
        selection!(set, children)

        updateDense!(set)
        return set
    end
end