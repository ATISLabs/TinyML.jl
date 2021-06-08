function evaluate!(set::TrainingSet, fitness::Function)
    Threads.@threads for cand in children(set)
        sum = 0.0
        for _ in 1:evals_per_candidate(set)
            sum += fitness(network(cand))
        end
        fitness!(cand, sum / evals_per_candidate(set))
    end
end