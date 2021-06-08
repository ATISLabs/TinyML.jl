module Bagging

    using ..EnsembleCore
    using ..Genetic
    using ..Flux
    using ..Misc

    export BaggingEnsemble, ensemble

    struct BaggingEnsemble
        ensembles::Ensemble
        backup::Ensemble
    end

    BaggingEnsemble(ensemble::Ensemble) = BaggingEnsemble([ensemble, deepcopy(ensemble)])

    @inline ensemble(bagging::BaggingEnsemble) = bagging.ensemble
    @inline backup(bagging::BaggingEnsemble) = bagging.backup

    @inline (bagging::BaggingEnsemble)(input::AbstractArray) = ensemble(bagging)(input)

    function Genetic.GeneticCore.randcopylayer(bagging::BaggingEnsemble)
        model = ensemble(bagging)
        weaks = classifiers(model)

        new_weaks = Vector{Chain}(undef, length(model))
        new_layers = Vector{Any}(undef, length(layers(weaks[1])))
        for (i, weak) in enumerate(weaks)
            for (j, layer) in enumerate(layers(weak))
                new_layers[j] = Genetic.GeneticCore.randcopylayer(layer)
            end
            new_weaks[i] = Chain(new_layers...)
        end

        BaggingEnsemble(Ensemble(new_weaks, model))
    end

    function Genetic.GeneticCore.mixweights!(set::TrainingSet, 
            child::BaggingEnsemble, 
            father::BaggingEnsemble, 
            mother::BaggingEnsemble)
        @inline classifier(bagging::BaggingEnsemble) = 
            classifiers(ensemble(bagging))[index!(bagging)]

        child_weak, father_weak, mother_weak = 
            classifier(child), classifier(father), classifier(mother)

        for (child_layer, father_layer, mother_layer) in
                zip(layers(child_weak), layers(father_weak), layers(mother_weak))
            Genetic.GeneticCore.mixweights!(set, child_layer, father_layer, mother_layer)
        end
    end

    function Genetic.GeneticCore.mutate!(set::TrainingSet,
            child::BaggingEnsemble)
        @inline classifier(bagging::BaggingEnsemble) = 
            classifiers(ensemble(bagging))[index(bagging)]

        for layer in classifier(child)
            Genetic.GeneticCore.mutate!(set, layer)
        end
    end

    function Genetic.GeneticCore.update!(old::BaggingEnsemble, new::BaggingEnsemble)
        for (oldweak, neweak) in zip(classifiers(ensemble(old)), classifiers(ensemble(new)))
            for (oldlayer, newlayer) in zip(layers(oldweak), layers(neweak))
                Genetic.GeneticCore.update!(oldlayer, newlayer)
            end
        end
    end

    # Displays 
    function Base.show(io::IO, bagging::BaggingEnsemble)
        print(io, "BaggingEnsemble($(ensemble(bagging)))")
    end

    function Base.display(io::IO, bagging::BaggingEnsemble)
        print(io, "BaggingEnsemble($(ensemble(bagging)))")
    end

end