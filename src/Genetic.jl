"""
    mutable struct TrainingSet

Stores all data for genetic algorithm.

# Structure
```julia
candidates::Array{<:Candidate, 1}
fitnessFunc::Function

popSize::Int64
selectedCandidatesCount::Int64
crossoverCuttingFractionDivisor::Int64
mutationRate::Float64
```
- candidates: Stores an array of candidates to be tested
- fitnessFunc: Stores the function that evaluates candidates
- popSize: Number of candidates
- selectedCandidatesCount: Number of candidates to be selected to the next generation
- crossoverCuttingFractionDivisor: Number used to find crossover point by splitting neurons
     weights on crossover (crossoverPoint = numberOfWeightsPerNeuron / crossoverCuttingFractionDivisor)
- mutationRate: Probability of mutation for each weight

# Initialization
```julia
TrainingSet(candidates::Array{<:Candidate, 1}, fitnessFunc::Function, populationSize::Int64,
            selectedCandidatesNumber::Int64, crossoverDivisor::Int64, mutationRate::Float64)
```

# Example
```julia_repl
julia> TrainingSet(snakeCandidates, snakeFitness, 2000, 10, 2, 0.1)
```
"""
mutable struct TrainingSet
    candidates::Array{<:Candidate, 1}
    fitnessFunc::Function

    popSize::Int64
    selectedCandidatesCount::Int64
    crossoverCuttingFractionDivisor::Int64
    mutationRate::Float64
end

"""
    genSwapArrayIndexes(candidates::Array{<:Candidate, 1}, index1::Int, index2::Int)

Swaps two elements posititons in candidates array.

# Example
```julia_repl
julia> genSwapArrayIndexes(candidatesArray, 1, 2)
```
This example will swap candidate 1 and candidate 2 positions. i.e 
candidate 1 will be at 2nd position and candidate 2 will be at 1st position.
"""
function genSwapArrayIndexes(arr::Array{<:Candidate, 1}, index1::Integer, index2::Integer)
    aux = arr[index1]
    arr[index1] = arr[index2]
    arr[index2] = aux
end

"""
    genSelectionRoulette(candidates::Array{<:Candidate, 1}, selectedCandidatesCount::Int)

Selects a specified number of candidates based on probability calculated using fitness values.

**Algorithm explanation:**
- It sorts the candidates array by fitness field reversively (e.g [5.0, 4.3, 3.0, 2.0, 1.0...])
- It sums all fitness values and calculates a probability for each cadidate using 'fitness/fitnessSum'.
- After, it walks through the array positions and doesn't stop until finds an element that satisfies the
following operation 'if array[i].probability > rand()', if this operation returns a 'true' value, the
algorithm will swap array element positions of the current index with one of the array starting indexes.
**The candidates will be at initial positions of the candidates array.**

# Example
```julia_repl
julia> genSelectionRoulette(candidates, 10)
```
Consider the example above is an array of 100 candidates, the steps made in this method
are these following:
- Reversively sort the array by fitness
- Loop 'i' until current numberOfSelectedCandidates < selectedCandidatesCount
    - Calculate fitness sum starting from 1 to end of the array
    - For each candidate calculate its probability by doind 'fitness / fitnessSum'
    - Chosen = rand()
    - Loop 'j' starting from 'i' to the end of the array 
        - find some candidate that satisfies candidates[j] > Chosen
            genSwapArrayIndexes(candidates, i, j)
- The selected candidates will be at initial positions
"""
function genSelectionRoulette(cands::Array{<:Candidate, 1}, selectedCandidatesCount::Integer)
    #reversively sorts the array by each item fitness
    sort!(cands, by=v->v.fitness, rev=true)

    for i in 1:selectedCandidatesCount
        arr = view(cands, i:length(cands))

        #calculates fitness sum
        fitSum = 0
        for cand in arr
            fitSum += cand.fitness
        end

        #calculates each candidate probability
        lastProb = 0
        for cand in arr
            cand.probability = cand.fitness / fitSum + lastProb
            lastProb = cand.probability
        end

        #selects one item from the list by a random value of probability
        chosen = rand()
        for j in eachindex(arr)
            if arr[j].probability > chosen
                genSwapArrayIndexes(cands, i, j + i - 1)
                break
            end
        end
    end
end

"""
    genSelectionBest(candidates::Array{<:Candidate, 1}, selectedCandidatesCount::Int)

Selects a specified number of candidates based on fitness values.

**Algorithm explanation:**
- It sorts the candidates array by fitness field reversively (e.g [5.0, 4.3, 3.0, 2.0, 1.0...])
**The selected candidates will be at initial positions of the candidates array.**

# Example
```julia_repl
julia> genSelectionRoulette(candidates, 10)
```
Consider the example above is an array of 100 candidates, the steps made in this method
are these following:
- Reversively sort the array by fitness
- The selected candidates will be at initial positions
"""
function genSelectionBest(cands::Array{<:Candidate, 1}, selectedCandidatesCount::Integer)
    sort!(cands, by=v->v.fitness, rev=true)
end

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
@inline function genGetCrossoverPoint(length::Int, crossoverDivisor::Int)
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
function genCrossoverMixParentsWeights(father::Float64NeuralNetwork, mother::Float64NeuralNetwork,
        child::Float64NeuralNetwork, crossoverDivisor::Int)
    for j in eachindex(child.layers)
        crossoverPoint = genGetCrossoverPoint(size(child.layers[j], 2), 
                                        crossoverDivisor)
        child.biases[j] = rand(Bool) ? father.biases[j] : mother.biases[j]

        neuronNumber = size(child.layers[j])[1]
        weightsNumber = size(child.layers[j])[2]

        fatherW = father.layers[j]
        motherW = mother.layers[j]
        childW = child.layers[j]
        for k in 1:neuronNumber
            for i in 1:crossoverPoint
                childW[k, i] = fatherW[k, i]
            end
            for i in crossoverPoint+1:weightsNumber
                childW[k, i] = motherW[k, i]
            end
        end
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
function genCrossoverMixParentsWeights(father::BitNeuralNetwork, mother::BitNeuralNetwork, 
        child::BitNeuralNetwork, crossoverDivisor::Int)
    for j in eachindex(child.layers)
        crossoverPoint = genGetCrossoverPoint(size(child.layers[j], 2), 
                                        crossoverDivisor)
        child.biases[j] = rand(Bool) ? father.biases[j] : mother.biases[j]
        neuronNumber = size(child.layers[j], 1)

        #performance reasons
        fChunks = father.layers[j].chunks
        mChunks = mother.layers[j].chunks
        cChunks = child.layers[j].chunks

        weightsNumber = size(cChunks)[2]
        limitStart = convert(Int64, floor(crossoverPoint / 64))

        rest = crossoverPoint % 64
        rest64 = 64 - rest
        if rest > 0
            for k in 1:neuronNumber
                limit = limitStart
                for i in 1:limit
                    cChunks[k, i] = fChunks[k, i]
                end
                
                limit += 1
                cChunks[k, limit] = ((fChunks[k, limit] << rest64) >> rest64) + 
                                            ((mChunks[k, limit] >> rest) << rest)

                for i in limit+1:weightsNumber
                    cChunks[k, i] = mChunks[k, i]
                end
            end
        else
            for k in 1:neuronNumber
                limit = limitStart
                for i in 1:limit
                    cChunks[k, i] = fChunks[k, i]
                end
                for i in limit+1:weightsNumber
                    cChunks[k, i] = mChunks[k, i]
                end
            end
        end
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
function genCrossover(cands::Array{<:Candidate, 1}, selectedCandidatesCount::Integer, 
        crossoverDivisor::Integer)
    Threads.@threads for i in (selectedCandidatesCount + 1):length(cands)
        father = cands[rand(1:selectedCandidatesCount)].net
        mother = cands[rand(1:selectedCandidatesCount)].net
        child = cands[i].net

        genCrossoverMixParentsWeights(father, mother, child, crossoverDivisor)
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
function genCrossoverClone(cands::Array{<:Candidate, 1}, selectedCandidatesCount::Integer, 
    crossoverDivisor::Integer)
    for i in selectedCandidatesCount:length(cands)
        parent = cands[rand(1:selectedCandidatesCount)]

        cands[i] = deepcopy(parent)
    end
end

"""
    genMutateWeight(weights::Array{Float64, 2}, neuron::Int, mutationRate::Float64)

Randomizes weights values for the weights of a specified neuron.

# Example
```julia_repl
julia> genMutateWeight(weights, 1, mutationRate)
```
Note: All weights are randomized in a range of -1 to 1 
"""
@inline function genMutateNet(net::Float64NeuralNetwork, mutationRate::Float64)
    for layer in net.layers
        for i in 1:size(layer, 1)
            for j in 1:size(layer, 2)
                if mutationRate > rand()
                    layer[i, j] = rand(Uniform(-1, 1))
                end
            end
        end
    end
end

"""
    genMutateWeight(weights::BinMatrix{2}, neuron::Int, mutationRate::Float64)

Randomizes weights values for the weights of a specified neuron.

# Example
```julia_repl
julia> genMutateWeight(weights, 1, mutationRate)
```
Note: All weights are randomized as 'true' or 'false' 
"""
@inline function genMutateNet(net::BitNeuralNetwork, mutationRate::Float64)
    for layer in net.layers
        for i in 1:size(layer, 1)
            for j in 1:size(layer, 2)
                if mutationRate > rand()
                    layer[i, j] = rand(Bool)
                end
            end
        end
    end
end

"""
    genMutation(cands::Array{<:Candidate, 1}, selectedCandidates::Int, mutationRate::Float64)

Randomize some weights of the not selected candidates based on a probability of 'mutationRate'.

# Example
```julia_repl
julia> genMutation(candidates, 10, mutationRate)
```
In the above example, consider a candidates array of 100 subjects.
The first 10 subjects of the array are the ones we assume that were selected before.
The method executes for the 90 rest subject neurons weights a mutation with random values based on
on 'mutationRate' as a probability of mutation.
"""
function genMutation(cands::Array{<:Candidate, 1}, selectedCandidates::Int, mutationRate::Float64)
    for i in selectedCandidates:length(cands)
        genMutateNet(cands[i].net, mutationRate)
    end
end

"""
    genExecute(tset::TrainingSet, genCount::Int)

Executes genetic algorithm on a pre-made training set for the number of generations specified
in 'genCount'.
To summarize, it resumes a training process.

# Example
```julia_repl
julia> genExecute(tset, 10)
```
In the example above, the genetic algorithm will execute the following steps:
- Loop generation until generation < genCount
    - select some candidates from population (functions e.g genSelectionBest(), genSelectionRoulette)
    - generate children based on selected candidates (functions e.g genCrossover(), genCrossoverClone())
    - mutate some of the children weights (functions e.g genMutation())
    - evaluate candidates
- Return training set
"""
function genExecute(tset::TrainingSet, genCount::Integer)
    cands = tset.candidates
    selCount = tset.selectedCandidatesCount
    cross = tset.crossoverCuttingFractionDivisor
    mutation = tset.mutationRate

    for gen in 1:genCount
        genSelectionBest(cands, selCount)
        genCrossover(cands, selCount, cross)
        genMutation(cands, selCount, mutation)
        
        Threads.@threads for i in 1:Threads.nthreads()
            fraction = floor(Int, length(cands) / Threads.nthreads())
            for j in (i-1)*fraction+1:(i)*fraction
                tset.fitnessFunc(cands[j])
            end
        end
    end

    sort!(tset.candidates, by=x->x.fitness, rev=true)

    return tset
end

"""
    genExecute(tset::TrainingSet, genCount::Int)

Generates a new training set and trains it for the number of generations specified
in 'genCount'.

# Example
```julia_repl
julia> genExecute(tset, 10)
```
In the example above, the genetic algorithm will execute the following steps:
- Generate some candidates
- Evaluate candidates
- Loop generation until generation < genCount
    - select some candidates from population (functions e.g genSelectionBest(), genSelectionRoulette)
    - generate children based on selected candidates (functions e.g genCrossover(), genCrossoverClone())
    - mutate some of the children weights (functions e.g genMutation())
    - evaluate candidates
- Return training set
"""
function genExecute(createCandidate::Function, fitnessFunc::Function, populationSize::Integer, 
        generationCount::Integer, selectedCandidatesCount::Integer, 
        crossoverCuttingFractionDivisor::Integer, mutationRate::Float64)
    candidates = [fitnessFunc(createCandidate()) for i in 1:populationSize]
    tset = TrainingSet(candidates, fitnessFunc, populationSize, selectedCandidatesCount, 
                    crossoverCuttingFractionDivisor, mutationRate)
    
    genExecute(tset, generationCount)

    return tset
end