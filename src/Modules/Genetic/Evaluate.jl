function evaluate!(set::TrainingSet)
    Threads.@threads for cand in set.candidates
        setFitness!(cand, set.fitnessFunc(getNetwork(cand)))
    end
end