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

    for i in 1:jlim
        for j in 1:crosspt
           childW[j,i] = fatherW[j,i] 
        end
    end
    for i in 1:jlim
        for j in crosspt+1:ilim
           childW[j,i] = motherW[j,i] 
        end
    end
    childB .= fatherB
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
    
    for i in 1:jlim
        for j in 1:crosspt
            @inbounds childW[j,i] = fatherW[j,i]
        end
    end
    for i in 1:jlim
        for j in crosspt+1:ilim
            @inbounds childW[j,i] = motherW[j,i]
        end
    end
    childB .= fatherB
end

function crossover!(set::TrainingSet)
    cands = getChildren(set)
    elitism = getBestPerformed(set)

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
    children = getChildren(set)
    elitism = getBestPerformed(set)
    Threads.@threads for i in 1:length(children)
        parent = rand(elitism)
        children[i] = Candidate(set, parent)
    end
end