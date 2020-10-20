function nextMove!(neatChain::Chain, data, 
        nextScene::Function, getInput::Function)
    in = getInput(data)
    act = neatChain(in)
    nextScene(data, act)

    in = @SVector in
    act = @SVector act

    return @SVector [in, act]
end

function generateSet!(data, nextScene::Function, getInput::Function,
        set::NEAT.TrainingSet, setSize::Int)
    neatChain = NEAT.NEATCore.getChain(set)
    tset = @SVector [
            nextMove!(neatChain, data, nextScene, getInput)
        for i in 1:setSize]

    return tset
end

function fitness(tset::SArray, chain::Chain)
    out = Array{Float64, 1}(undef, length(tset))

    for (i, row) in enumerate(tset)
        out[i] = mean(abs.(chain(row[1]) .- row[2]))
    end

    return out
end

function randomConverter!(chain::Chain,
        layers::Tuple,
        possibleInputs::Array{Tuple{Int64, Int64}, 1}, 
        set::NEAT.TrainingSet;
        setSize::Int=100)
    tset = generateSet(possibleValues, set, setSize)
    randomTrain!(tset)
end