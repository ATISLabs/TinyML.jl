module NEAT
    using Reexport
    using ..BitFlux
    using ..Misc

    module NEATCore
        #= Libraries =#
        using Distributions: Uniform
        using Flux
        using ..Misc

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
            children::Array{Candidate} = []
            if isTrained!(set)
                children = crossover!(set)
                mutation!(set, children)
            else
                children = getFirstChildren(set)
                evaluate!(set, children)
            end

            gen = 0
            while gen < genNumber && 
                    getFitness(unsafeGetRepresentant(set.species[1])) < maxFitness
                println("Gen: $(gen) -- Fitness: ",
                    "$(getFitness(unsafeGetRepresentant(set.species[1])))")
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