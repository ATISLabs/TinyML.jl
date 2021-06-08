module Genetic
    using Reexport
    using ..BitFlux
    using ..Misc

    module GeneticCore
        #= Imports =#
        using Flux
        using Distributions: Uniform

        #= Upper import =#
        using ..BitFlux
        using ..Misc

        #= Includes =#
        include("./Genetic/Structs/Candidate.jl")
        include("./Genetic/Structs/TrainingSet.jl")
        include("./Genetic/Base.jl")
        include("./Genetic/Selection.jl")
        include("./Genetic/Crossover.jl")
        include("./Genetic/Mutation.jl")
        include("./Genetic/Evaluate.jl")

        #= Exports =#
        export train!, TrainingSet, Candidate, fitness, fitness!,
            network, elitism, elitism!, evals_per_candidate, evals_per_candidate!,
            crossover_divisor, crossover_divisor!, mutation_rate, mutation_rate!, 
            children, best, best!, gen, bestfitness, crossover_clone!, selection_best!,
            mutation_rand!, evaluate!
    end

    @reexport using .GeneticCore
end