module SetController
    using ..AI

    export createSet, saveSetToFile, loadSetFromFile, deleteSetFile

    const TSET_NAME = "tset.jld2"

    function createSet(x::NetworkType, args...)
        if x == FloatMLP
            return createFloatGeneticSet(args...)
        elseif x == BitMLP
            return createFloatGeneticSet(args...)
        elseif x == FloatNEAT
            return createFloatNEATSet(args...)
        elseif x == BitNEAT
            #f == aiGetBitSensorsData
        elseif x == FloatCNN
            #f = aiGetFloatDrawingMatrix
        else
            #f = aiGetBitDrawingMatrix
        end
    end

    saveSetToFile(arr::Union{NEAT.TrainingSet,Genetic.TrainingSet}) = @save TSET_NAME arr

    function loadSetFromFile()
        arr = :TrainingSet

        @load TSET_NAME arr
        
        return arr
    end

    deleteSetFile() = rm(TSET_NAME)
end