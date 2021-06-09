function crossovermix!(set::TrainingSet)
    cands = children(set)
    elitism = best(set)

    for child in cands
        father = network(rand(elitism)).layers
        mother = network(rand(elitism)).layers
        childl = network(child).layers
        
        for j in indexes(set)
            mixweights!(set, father[j], mother[j], childl[j])
        end
    end
end

function crossover_clone!(set::TrainingSet)
    cands = children(set)
    elitism = best(set)
    for i in 1:length(cands)
        parent = rand(elitism)
        cands[i] = Candidate(set, parent)
    end
end

@inline function crossover_point(length::Int, crossover_divisor::Int)
    return convert(Int64, floor(length / crossover_divisor))
end

function mixweights!(set::TrainingSet, 
        father::Dense, mother::Dense, child::Dense)
    ilim = size(weight(father), 1)
    crosspt = crossover_point(ilim, crossover_divisor(set))
    jlim = size(weight(father), 2)

    fatherW = weight(father)
    motherW = weight(mother)
    childW = weight(child)

    fatherB = bias(father)
    motherB = bias(mother)
    childB = bias(child)

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

function mixweights!(set::TrainingSet, 
        father::BitDense, mother::BitDense, child::BitDense)
    fatherW = chunks(weight(father))
    motherW = chunks(weight(mother))
    childW = chunks(weight(chunks))

    ilim = size(fatherW, 1)
    crosspt = crossover_point(ilim, crossover_divisor(set))
    jlim = size(fatherW, 2)

    fatherB = bias(father)
    motherB = bias(mother)
    childB = bias(child)
    
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
