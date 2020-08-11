@inline function getCrossoverPoint(length::Int, crossoverDivisor::Int)
    return convert(Int64, floor(length / crossoverDivisor))
end

function crossoverMixParentsWeights!(set::TrainingSet, 
        father::Dense, mother::Dense, child::Dense)
    ilim = size(father.W, 1)
    crosspt = getCrossoverPoint(ilim, getCrossoverDivisor(set))
    jlim = size(father.W, 2)

    fatherW = father.W
    motherW = mother.W
    childW = child.W

    fatherB = father.b
    motherB = mother.b
    childB = child.b

    for i in 1:crosspt
        for j in 1:jlim
           childW[i,j] = fatherW[i,j] 
        end
        childB[i] = fatherB[i]
    end
    for i in crosspt+1:ilim
        for j in 1:jlim
           childW[i,j] = motherW[i,j] 
        end
        childB[i] = motherB[i]
    end
end

function crossoverMixParentsWeights!(set::TrainingSet, 
        father::BitDense, mother::BitDense, child::BitDense)
    fatherW = father.W.chunks
    motherW = mother.W.chunks
    childW = child.W.chunks

    ilim = size(fatherW, 1)
    crosspt = getCrossoverPoint(ilim, getCrossoverDivisor(set))
    jlim = size(fatherW, 2)

    fatherB = father.b
    motherB = mother.b
    childB = child.b
    

    for i in 1:crosspt
        for j in 1:jlim
            @inbounds childW[i,j] = fatherW[i,j]
        end
        @inbounds childB[i] = fatherB[i]
    end
    for i in crosspt+1:ilim
        for j in 1:jlim
            @inbounds childW[i,j] = motherW[i,j]
        end
        @inbounds childB[i] = motherB[i]
    end
end

function crossover!(set::TrainingSet)
    cands = view(getCandidates(set), (getElitism(set)+1):length(getCandidates(set)))
    elitism = view(getCandidates(set), 1:getElitism(set))

    Threads.@threads for child in cands
        father = getNetwork(rand(elitism)).layers
        mother = getNetwork(rand(elitism)).layers
        childl = getNetwork(child).layers
        
        for j in set.indexes
            crossoverMixParentsWeights!(set, father[j], mother[j], childl[j])
        end
    end
end

function crossoverClone!(set::TrainingSet)
    cands = view(getCandidates(set), getElitism(set)+1:length(getCandidates(set)))
    elitism = view(getCandidates(set), 1:getElitism(set))
    Threads.@threads for i in 1:length(cands)
        parent = rand(elitism)
        cands[i] = Candidate(set, parent)
    end
end