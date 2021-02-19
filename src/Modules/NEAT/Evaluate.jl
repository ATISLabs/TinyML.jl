function evaluate!(set::TrainingSet, children::Array{Candidate,1})
    Threads.@threads for thread in 0:(Threads.nthreads()-1)
        tChildren = copy(children)
        chain, index = getEvaluationChain(set)
        fitnessFunc = set.fitnessFunc

        start = ceil(Int, thread * length(tChildren) / Threads.nthreads()) + 1
        stop = (start <= length(children)) * ceil(Int, (thread+1) * length(tChildren) / Threads.nthreads())
        data = deepcopy(getAdditionalData(set))

        for i in start:stop
            fitness = 0.0
            unsafeReplaceCandidate!(chain, index, tChildren[i])
            for j in 1:getEvalsPerCandidate(set)
                fitness += fitnessFunc(chain, data)
            end
            tChildren[i].fitness = fitness / getEvalsPerCandidate(set)
        end
    end
end