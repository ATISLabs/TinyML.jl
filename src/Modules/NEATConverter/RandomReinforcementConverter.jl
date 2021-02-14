#random(in) = rand(Float32, in)
random(in) = [3.f0,2.f0]

bitsConverter(arr::Array{Float32,1}) = arr .> 0

function generateSet(layer::NEATDense, limit::Int)
    in = NEAT.NEATCore.getInputNumber(layer)

    dataset = Array{Pair{BitArray{1}, Array{Float32, 1}}, 1}(undef, limit)

    for i in 1:limit
        row = random(in)
        bitsRow = bitsConverter(row)
        @inbounds dataset[i] = bitsRow => layer(row)
    end
    dataset
end

function fitness(net::Chain, 
        dataset::Array{Pair{BitArray{1}, Array{Float32, 1}}})
    out = 0
    for row in dataset
        netOutput = net(row[1])
        temp = 0
        for j in 1:length(netOutput)
            @inbounds temp += abs(netOutput[j] - row[2][j])
        end
        out += 1 / (temp / length(netOutput))
    end
    out / length(dataset)
end

function convert!(net::Chain, layersToTrain::Tuple, neatLayer::NEATDense, 
        ; samples::Int=100, 
        maxFitness::Number=typemax(Float64), 
        genNumber::Int=typemax(Int))
    set = Genetic.TrainingSet(net, layersToTrain, fitness, 
            data=generateSet(neatLayer, samples))
    Genetic.train!(set, genNumber=genNumber, maxFitness=maxFitness)
end