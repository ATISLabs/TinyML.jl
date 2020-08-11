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
function swapArrayIndexes!(arr::Array{Candidate, 1}, index1::Integer, index2::Integer)
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
function selectionBest!(tset::TrainingSet)
    sort!(tset.candidates, by=v->getFitness(v), rev=true)
end
