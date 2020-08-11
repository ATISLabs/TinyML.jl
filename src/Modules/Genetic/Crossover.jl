"""
    genGetCrossoverPoint(layer, crossoverDivisor::Int)

Returns the crossover point based on how many weights are in neuron of the layer and
a the specified divisor.

# Example
Consider the neurons of this layer have 20 weights each
```julia_repl
julia> genCrossoverPoint(layer, 2)
10
```
"""
@inline function getCrossoverPoint(length::Int, crossoverDivisor::Int)
    return convert(Int64, floor(length / crossoverDivisor))
end

"""
    genCrossoverMixParentsWeights(father::Float64NeuralNetwork, mother::Float64NeuralNetwork,
                child::Float64NeuralNetwork, crossoverDivisor::Int)

Copies a fraction of weights of the father weights to the child weights, and the same is done
with the mother weights.

# Example
```julia_repl
julia> genCrossoverMixParentsWeights(layerFromFather, layerFromMother, layerFromChild, 2)
```
Consider the example above is a receiving the respective layers from father, mother and child,
and the layers have 100 neurons with 20 weights each:
- For each neuron of each layer do 'child[neuronIndex, i] = father[neuronIndex, i]' until 'i' > crossoverPoint
- For each neuron of each layer do 'child[neuronIndex, i] = father[neuronIndex, i]', 'i' starting from
        crossoverPoint to number of weights
"""
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

"""
    genCrossoverMixParentsWeights(father::BitNeuralNetwork, mother::BitNeuralNetwork,
                child::BitNeuralNetwork, crossoverDivisor::Int)

Copies a fraction of weights of the father weights to the child weights, and the same is done
with the mother weights.

# Example
```julia_repl
julia> genCrossoverMixParentsWeights(layerFromFather, layerFromMother, layerFromChild, 2)
```
Consider the example above is a receiving the respective layers from father, mother and child,
and the layers have 100 neurons with 20 weights each:
- crossoverPoint = floor(crossoverPoint / 64)
- For each neuron of each layer do 'child[neuronIndex, i] = father[neuronIndex, i]' until 'i' > crossoverPoint
- rest = crossoverPoint % 64
//The step under is used to merge father and mother when crossover point is not a full chunk, i.e when it splits
//at bit level
- if rest > 0 then
    - crossoverPoint = crossoverPoint + 1
    - child[neuronIndex, crossoverPoint] = ((father[neuronIndex, crossoverPoint] << (rest-64)) >> rest-64)+
                                            ((father[neuronIndex, crossoverPoint] << rest) >> rest)
    //The step above is used to set the not required father fraction to zeros and the same to mother fraction
    //father 10101|10 => 10000|00 => 00000|10 => 00000|10 => 00000|10
    //mother 10001|01 => 10001|01 => 10001|01 => 00100|01 => 10001|00
    //child                                               => 10001|10
- For each neuron of each layer do 'child[neuronIndex, i] = father[neuronIndex, i]', 'i' starting from
        crossoverPoint+1 to number of weights
"""
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

"""
    genCrossover(candidates::Array{<:Candidate, 1}, selectedCandidatesCount::Int)

Generate new candidates by mixing father and mother weights.

# Example
```julia_repl
julia> genCrossover(candidates, 10, 2)
```
In the example above we assume the first positions, limited by the variable 'selectedCandidatesCount'
are the selected candidates.
Consider for the example the number of candidates is 100, the steps executed by the method are:
- for each candidate starting from candidates[selectedCandidatesCount + 1] to end of array, do
    - choose a random father from selected ones
    - choose a random mother from selected ones
    - Loop 'i' for the number of layers in the child
        crossoverPoint = genGetCrossoverPoint(layer)
        - for 'j' for the number of neurons in the layer 
            genCrossoverMixParentsWeights(father.layers[j], mother.layers[j], child.layers[j],
                                crossoverPoint, neuronIndex = j)
"""
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

"""
    genCrossoverClone(candidates::Array{<:Candidate, 1}, selectedCandidatesCount::Int)

Generate a new candidate by cloning a random candidate from selected ones.

# Example
```julia_repl
julia> genCrossoverClone(candidates, 10)
```
In the example above we assume the first positions, limited by the variable 'selectedCandidatesCount'
are the selected candidates.
Consider for the example the number of candidates is 100, the steps executed by the method are:
- for each candidate starting from candidates[selectedCandidatesCount + 1] to end of array, do
    - select a random candidate from selected ones
    - candidate = copy(randomlySelected)
"""
function crossoverClone!(set::TrainingSet)
    cands = view(getCandidates(set), getElitism(set)+1:length(getCandidates(set)))
    elitism = view(getCandidates(set), 1:getElitism(set))
    Threads.@threads for i in 1:length(cands)
        parent = rand(elitism)
        cands[i] = Candidate(set, parent)
    end
end