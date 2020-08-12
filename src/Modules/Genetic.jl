module Genetic
    #= Imports =#
    using Flux
    using Distributions: Uniform

    #= Local imports =#

    #= Upper import =#
    using ..BitFlux

    #= Includes =#
    include("./Genetic/Base.jl")
    include("./Genetic/Selection.jl")
    include("./Genetic/Crossover.jl")
    include("./Genetic/Mutation.jl")
    include("./Genetic/Evaluate.jl")

    #= Exports =#
    export Train!, getFitness, setFitness!, TrainingSet, getChain

    
    function Train!(tset::TrainingSet, genCount::Integer)
        evaluate!(tset)
        for gen in 1:genCount
            selectionBest!(tset)
            crossover!(tset)
            mutation!(tset)
            evaluate!(tset)
        end

        updateChain!(tset)

        return tset
    end

end