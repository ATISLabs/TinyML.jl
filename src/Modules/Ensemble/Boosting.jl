module Boosting

    using Flux
    using StatsBase
    using ..EnsembleCore
    using ..Genetic
    using ..Misc

    export BoostingEnsemble, ensemble

    struct BoostingEnsemble
        ensemble::Ensemble
        index::Ref{Int}
    end

    BoostingEnsemble(ensemble::Ensemble) = BoostingEnsemble(ensemble, Ref(1))

    @inline ensemble(boosting::BoostingEnsemble) = boosting.ensemble
    @inline indexref(boosting::BoostingEnsemble) = boosting.index
    @inline index(boosting::BoostingEnsemble) = boosting.index[]
    @inline index!(boosting::BoostingEnsemble) = 
        if index(boosting) <= length(ensemble(boosting)) 
        boosting.index[] += 1 else boosting.index[] = 1 end

    @inline (boosting::BoostingEnsemble)(input::AbstractArray) = ensemble(boosting)(input)

    function Genetic.GeneticCore.randcopylayer(boosting::BoostingEnsemble)
        model = ensemble(boosting)
        weaks = classifiers(model)

        new_weaks = Vector{Chain}(undef, length(model))
        new_layers = Vector{Any}(undef, length(layers(weaks[1])))
        for (i, weak) in enumerate(weaks)
            for (j, layer) in enumerate(layers(weak))
                new_layers[j] = Genetic.GeneticCore.randcopylayer(layer)
            end
            new_weaks[i] = Chain(new_layers...)
        end

        BoostingEnsemble(Ensemble(new_weaks, model), indexref(boosting))
    end

    function Genetic.GeneticCore.mixweights!(set::TrainingSet, 
            child::BoostingEnsemble, 
            father::BoostingEnsemble, 
            mother::BoostingEnsemble)
        @inline classifier(boosting::BoostingEnsemble) = 
            classifiers(ensemble(boosting))[index(boosting)]

        child_weak, father_weak, mother_weak = 
            classifier(child), classifier(father), classifier(mother)

        for (child_layer, father_layer, mother_layer) in
                zip(layers(child_weak), layers(father_weak), layers(mother_weak))
            Genetic.GeneticCore.mixweights!(set, child_layer, father_layer, mother_layer)
        end
    end

    function Genetic.GeneticCore.mutate!(set::TrainingSet,
            child::BoostingEnsemble)
        @inline classifier(boosting::BoostingEnsemble) = 
            classifiers(ensemble(boosting))[index(boosting)]

        for layer in classifier(child)
            Genetic.GeneticCore.mutate!(set, layer)
        end
    end

    function Genetic.GeneticCore.update!(old::BoostingEnsemble, new::BoostingEnsemble)
        for (oldweak, neweak) in zip(classifiers(ensemble(old)), classifiers(ensemble(new)))
            for (oldlayer, newlayer) in zip(layers(oldweak), layers(neweak))
                Genetic.GeneticCore.update!(oldlayer, newlayer)
            end
        end
    end

    function generateloop(boosting::BoostingEnsemble,
            dataset::Vector{Pair{X, Y}},
            checkprediction::Function; 
            samplefraction::Union{Int, Real}=0.6,
            iterations_per_classifier::Int=30) where {X, Y}
        n = length(dataset)
        step = 1.0/n
        weights = Weights(fill(0.5, n))

        function reweight!(set::TrainingSet)
            best = network(first(set))
            x(i) = dataset[i][1]
            y(i) = dataset[i][2]

            for i in 1:n 
                if checkprediction(best(x(i)), y(i))
                    weights[i] = max(weights[i] - step, 0.0)
                else
                    weights[i] = min(weights[i] + step, 1.0)
                end
            end
        end

        function generatefitness(dataset::Vector{Pair{X, Y}}, score::Function) where {X, Y}
            function fitness(net::Chain)
                scores = 0

                for entry in dataset
                    scores += score(net(entry[1]), entry[2])
                end

                scores
            end
            fitness
        end

        if samplefraction isa Real
            fraction = ceil(Int, n * samplefraction)
        else
            fraction = samplefraction
        end

        loopcounter = 0
        function loop!(set::TrainingSet, loss::Function)
            evaluate!(set, generatefitness(
                sample(dataset, weights, fraction), loss))
            selection_best!(set)
            reweight!(set)
            crossover_clone!(set)
            mutation_rand!(set)
            loopcounter += 1
            if loopcounter == iterations_per_classifier index!(boosting) end
        end

        loop!
    end

    # Displays 
    function Base.show(io::IO, boosting::BoostingEnsemble)
        print(io, "BoostingEnsemble($(ensemble(boosting)))")
    end

    function Base.display(io::IO, boosting::BoostingEnsemble)
        print(io, "BoostingEnsemble($(ensemble(boosting)))")
    end

end