module Genetic
    using Reexport
    using ..BitFlux

    module GeneticCore
        #= Imports =#
        using Flux
        using Distributions: Uniform

        #= Upper import =#
        using ..BitFlux

        #= Includes =#
        include("./Genetic/Base.jl")
        include("./Genetic/Selection.jl")
        include("./Genetic/Crossover.jl")
        include("./Genetic/Mutation.jl")
        include("./Genetic/Evaluate.jl")

        #= Exports =#
        export train!, TrainingSet
        
        function train!(tset::TrainingSet; 
                genNumber::Integer=typemax(Int64), maxFitness::Float64=Inf64)
            if !isTrained!(tset)
                evaluate!(tset)
            end
            gen = 0
            while gen < genNumber && 
                    getFitness(unsafeGetBest(tset)) < maxFitness
                println("Gen: $(gen) -- Fitness: $(getFitness(unsafeGetBest(tset)))")
                selectionBest!(tset)
                crossover!(tset)
                mutation!(tset)
                evaluate!(tset)
                gen += 1
            end

            updateChain!(tset)

            return gen
        end
    end

    @reexport using .GeneticCore
end