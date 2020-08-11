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
    export Train!, getFitness, setFitness!, TrainingSet

    """
        genExecute(tset::TrainingSet, genCount::Int)

    Executes genetic algorithm on a pre-made training set for the number of generations specified
    in 'genCount'.
    To summarize, it resumes a training process.

    # Example
    ```julia_repl
    julia> genExecute(tset, 10)
    ```
    In the example above, the genetic algorithm will execute the following steps:
    - Loop generation until generation < genCount
        - select some candidates from population (functions e.g genSelectionBest(), genSelectionRoulette)
        - generate children based on selected candidates (functions e.g genCrossover(), genCrossoverClone())
        - mutate some of the children weights (functions e.g genMutation())
        - evaluate candidates
    - Return training set
    """
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