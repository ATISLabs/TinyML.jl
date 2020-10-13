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
    export train!, getFitness, setFitness!, TrainingSet, getChain

    
    function train!(tset::TrainingSet; 
            genNumber::Integer=typemax(Int64), maxFitness::Float64=Inf64)
        if !isTrained(tset)
            evaluate!(tset)
        end
        gen = 0
        while gen < genNumber && 
                unsafeGetBest(tset) < maxFitness
            selectionBest!(tset)
            crossover!(tset)
            mutation!(tset)
            evaluate!(tset)
            gen += 1
        end

        updateChain!(tset)

        return (tset, gen)
    end

end