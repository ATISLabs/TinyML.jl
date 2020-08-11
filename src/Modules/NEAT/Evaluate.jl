function evaluate!(set::TrainingSet, children::Array{Network,1})
    Threads.@threads for thread in 0:(Threads.nthreads()-1)
        tChildren = copy(children)
        chain, index = getEvaluationChain(set)
        fitnessFunc = set.fitnessFunc

        start = ceil(Int, thread * length(tChildren) / Threads.nthreads()) + 1
        stop = (start <= length(children)) * ceil(Int, (thread+1) * length(tChildren) / Threads.nthreads())

        for i in start:stop
            unsafeReplaceNetwork!(chain, index, tChildren[i])
            tChildren[i].fitness = fitnessFunc(chain)
        end
    end
end