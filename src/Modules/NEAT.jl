module NEAT
    using Reexport
    using ..BitFlux

    module NEATCore
        #= Libraries =#
        using Distributions: Uniform
        using Flux

        #= Upper Imports =#
        using ..BitFlux

        #= Includes =#
        include("./NEAT/Base.jl")
        include("./NEAT/Forward.jl")
        include("./NEAT/Evaluate.jl")
        include("./NEAT/Crossover.jl")
        include("./NEAT/Selection.jl")
        include("./NEAT/Mutation.jl")

        export NEATDense, TrainingSet, train!

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
            return gen
        end
    end
    
    @reexport using .NEATCore
end