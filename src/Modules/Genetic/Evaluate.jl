function evaluate!(set::TrainingSet)
    Threads.@threads for cand in getChildren(set)
        avg = 0.0
        for i in 1:getEvalsPerCandidate(set)
            avg += getFitnessFunction(set)(getNetwork(cand))
        end
        setFitness!(cand, avg / getEvalsPerCandidate(set))
    end
end