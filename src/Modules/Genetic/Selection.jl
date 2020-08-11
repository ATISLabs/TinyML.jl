function swapArrayIndexes!(arr::Array{Candidate, 1}, index1::Integer, index2::Integer)
    aux = arr[index1]
    arr[index1] = arr[index2]
    arr[index2] = aux
end

#=function selectionRoulette!(set::TrainingSet)
    #reversively sorts the array by each item fitness
    cands = getCandidates(set)
    elitism = getElitism(set)

    sort!(cands, by=v->getFitness(v), rev=true)

    for i in 1:elitism
        arr = view(cands, i:length(cands))

        #calculates fitness sum
        fitSum = 0
        for cand in arr
            fitSum += getFitness(cand)
        end

        #calculates each candidate probability
        lastProb = 0
        for cand in arr
            cand.probability = getFitness(cand) / fitSum + lastProb
            lastProb = cand.probability
        end

        #selects one item from the list by a random value of probability
        chosen = rand()
        for j in eachindex(arr)
            if arr[j].probability > chosen
                swapArrayIndexes!(cands, i, j + i - 1)
                break
            end
        end
    end
end=#

function selectionBest!(tset::TrainingSet)
    sort!(tset.candidates, by=v->getFitness(v), rev=true)
end
