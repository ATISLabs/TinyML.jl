"""
    mutable struct Float64NeuralNetwork <: NeuralNetwork

Stores the layers and biases for using on Neural Network methods.
It uses only Float64 values.

# Structure
```julia
biases::Array{Float64, 1}
layers::Array{Array{Float64, 2}, 1}
```

# Initialization
```julia
Float64NeuralNetwork(inputsNumber::Int, hiddenLayersArchitecture::Array{Int, 1}, outputNeuronsNumber::Int)
```
- inputsNumber: Relative to the number of weights of the first effective layer
- hiddenLayersArchitecture: Relative to the number of hidden layers and how many 
neurons shall each layer have in
- outputNeuronsNumber: Relative to the number of neurons into the output layer

# Example
```julia_repl
julia> Float64NeuralNetwork(20, [20, 15], 5)
```
It will make a new Float64NeuralNetwork object, with 3 layers
and all of them having respectively, 20 neurons on the first
hidden layer, 15 neurons on the second hidden layer and 5 neurons
on the output layer
"""
mutable struct Float64NeuralNetwork <: MLPNeuralNetwork
    biases::Array{Float64, 1}
    layers::Array{Array{Float64, 2}, 1}

    function Float64NeuralNetwork(inputNumber::Int,
        hiddenLayersArchitecture::Array{Int, 1}, outputNeuronsNumber::Int)
            net = new()

            net.layers = Array{Array{Float64, 2}, 1}(undef, 0)
            net.biases = Array{Float64, 1}(undef, 0)
            neuronNumber = inputNumber
            for layer in hiddenLayersArchitecture
                push!(net.layers, rand(Uniform(-1, 1), layer, neuronNumber))
                push!(net.biases, rand(Float64))
                neuronNumber = layer
            end
            push!(net.layers, rand(Uniform(-1, 1), outputNeuronsNumber, neuronNumber))
            push!(net.biases, rand(Float64))

            return net
    end
end

"""
    mutable struct BitNeuralNetwork <: NeuralNetwork

Stores the layers and biases for using on Neural Network methods.
It uses only Bit values.

# Structure
```julia
biases::BitArray{1}
layers::Array{BinMatrix, 1}
```
- inputsNumber: Relative to the number of weights of the first effective layer
- hiddenLayersArchitecture: Relative to the number of hidden layers and how many 
neurons shall each layer have in
- outputNeuronsNumber: Relative to the number of neurons into the output layer

# Initialization
```julia
BitNeuralNetwork(inputsNumber::Int, hiddenLayersArchitecture::Array{Int, 1}, outputNeuronsNumber::Int)
```

# Example
```julia_repl
julia> Float64NeuralNetwork(20, [20, 15], 5)
```
It will make a new Float64NeuralNetwork object, with 3 layers
and all of them having respectively, 20 neurons on the first
hidden layer, 15 neurons on the second hidden layer and 5 neurons
on the output layer
"""
mutable struct BitNeuralNetwork <: MLPNeuralNetwork
    biases::BitArray{1}
    layers::Array{BinMatrix{2}, 1}

    function BitNeuralNetwork(inputNumber::Int,
        hiddenLayersArchitecture::Array{Int, 1}, outputNeuronsNumber::Int)
            net = new()

            net.layers = Array{BinMatrix{2}, 1}(undef, 0)
            net.biases = BitArray{1}(undef, 0)
            inputNumber = inputNumber
            for layer in hiddenLayersArchitecture
                push!(net.layers, BinMatrix{2}(undef, layer, inputNumber))
                push!(net.biases, rand(Bool))
                inputNumber = layer;
            end

            push!(net.layers, BinMatrix{2}(undef, outputNeuronsNumber, inputNumber))
            push!(net.biases, rand(Bool))

            return net
    end
end

#=
function nnetExecuteBLASLayer(layer::Array{Float64, 2}, input::Array{Float64, 1}, bias::Float64)
    return activationSigmoid.((layer * input) .+ bias)
end
=#

"""
    nnetExecuteLayer(layer::Array{Float64, 2}, input::Array{Float64, 1}, bias::Float64)

Returns a float64 array of values, where each line represents respectively an output of a neuron
(each neuron is a line of the 'layer' matrix).
It multiplies the matrix 'layer' with the vector 'input', and sums 'bias' in each element
, after that, it applies sigmoid function, which will give the return data

# Example
```julia_repl
julia> nnetExecuteLayer([[1.0, 2.0] [4.0, 3.0]], [1.0, 7.0], 3.0)
2-element Array{Float64, 1}
0.9999999999999873
0.999999999994891
```
In this example, the method is doing the following operations
```
1.0*1.0 + 2.0*7.0 + 3.0 => activationSigmoid(18) = 0.9999999999999873
4.0*1.0 + 3.0*7.0 + 3.0 => activationSigmoid(28) = 0.999999999994891
```
"""
function nnetExecuteLayer(layer::Array{Float64, 2}, input::Array{Float64, 1}, bias::Float64)
    neuronNumber = size(layer)[1]
    weightsNumber = size(layer)[2]
    arr = Array{Float64, 1}(undef, neuronNumber)

    for i in 1:neuronNumber
        el = 0
        for j in 1:weightsNumber
            el += layer[i, j] * input[j]
        end
        arr[i] = activationSigmoid(el + bias)
    end

    return arr
end

"""
    nnetExecuteLayer(layer::BinMatrix, input::BitArray{1}, bias::Bool)

Returns a bit array of values, where each line represents respectively an output of a neuron
(each neuron is a line of the 'layer' matrix).

For this method, consider 'falses' as 1 and 'trues' as 0
Also, consider a chunk as a UInt64, where the bits are stored.

**Explanation, steps executed on this method:**
- For each neuron (line of BinMatrix), it will apply a xor on its chunks which will result
in a new array
- Count the number of ones of this last calculated array
- Do '(ones + bias) > ((weightsPerNeuron + !bias) - ones)' and store it on the output bit array

On this way, all the output values will be like bits (true or false), and checking if the number of ones
is bigger than the number of zeros (total - ones) will be the same as verifying if the sum is positive
or negative (Remember, consider 'falses' as 1 and 'trues' as 0).

# Example
For this example, consider the BinMatrix being [[1, 0], [0, 1]]
```julia_repl
julia> nnetExecuteLayer(BinMatrix{2}(2, 2), BitArray([1, 1]), false)
2-element BitArray{1}
0
0
```
In this example, the method is doing the following operations
```
xor([1,0], [1,1]) => count_ones([0,1]) => (1+0) > ((2 + 1) - 1) = 0
xor([0,1], [1,1]) => count_ones([1,0]) => (1+0) > ((2 + 1) - 1) = 0
```
"""
function nnetExecuteLayer(layer::BinMatrix{2}, input::BitArray{1}, bias::Bool)
    neuronNumber = size(layer)[1]
    arr =BitArray{1}(undef, neuronNumber)
    inputNumberBits = size(layer)[2]

    chunkNumber = floor(Int, inputNumberBits / 64)
    chunkRestShift = 64 - inputNumberBits % 64

    chunks = layer.chunks
    inputChunks = input.chunks
    lastChunkIndex = size(chunks)[2]

    for i in 1:neuronNumber
        ones = 0
        for j in 1:chunkNumber
            ones += count_ones(xor(chunks[i, j], inputChunks[j]))
        end
        ones += count_ones((xor(chunks[i, lastChunkIndex], inputChunks[lastChunkIndex]) << 
                                chunkRestShift) >> chunkRestShift)
        arr[i] = (ones + bias) > ((inputNumberBits + !bias) - ones)
    end

    return arr
end

"""
    nnetExecuteOutputLayer(layer::BinMatrix{2}, input::BitArray{1}, bias::Bool)

Returns a float32 array of values, where each line represents respectively an output of a neuron
(each neuron is a line of the 'layer' matrix).

For this method, consider 'falses' as 1 and 'trues' as 0
Also, consider a chunk as a UInt64, where the bits are stored.

**Explanation, steps executed on this method:**
- For each neuron (line of BinMatrix), it will apply a xor on its chunks which will result
in a new array
- Count the number of ones of this last calculated array
- Do '1 - (ones + bias) / numberOfNeuronWeights' and store it on the output bit array

Differently of 'nnetExecuteLayer', this method returns an array of float32 values representing
the proportion of zeros by the number of weights of each element

# Example
For this example, consider the BinMatrix being [[1, 0], [0, 1]]
```julia_repl
julia> nnetExecuteLayer(BinMatrix{2}(2, 2), BitArray([1, 1]), false)
2-element BitArray{1}
0.5
0.5
```
In this example, the method is doing the following operations
```
xor([1,0], [1,1]) => count_ones([0,1]) => 1 - (1 + 0) / 2 = 0.5
xor([0,1], [1,1]) => count_ones([1,0]) => 1 - (1 + 0) / 2 = 0.5
```
"""
function nnetExecuteOutputLayer(layer::BinMatrix{2}, input::BitArray{1}, bias::Bool)
    neuronNumber = size(layer)[1]
    arr = Array{Float32, 1}(undef, neuronNumber)
    inputNumberBits = size(layer)[2]

    chunkNumber = floor(Int, inputNumberBits / 64)
    chunkRestShift = 64 - inputNumberBits % 64

    chunks = layer.chunks
    inputChunks = input.chunks
    lastChunkIndex = size(chunks)[2]

    for i in 1:neuronNumber
        ones = 0
        for j in 1:chunkNumber
            ones += count_ones(xor(chunks[i, j], inputChunks[j]))
        end
        ones += count_ones((xor(chunks[i, lastChunkIndex], inputChunks[lastChunkIndex]) << 
                                chunkRestShift) >> chunkRestShift)
        arr[i] = 1 - (ones + bias) / inputNumberBits
    end

    return arr
end

"""
    nnetExecute(net::NeuralNetwork, input::Array{<: Number, 1})

It is the neural network forward

Executes neuron operations through all network layers and returns the
output of the last layer

# Example
For this example, consider all the weights are 1 and all biases are 1
```julia_repl
julia> nnetExecute(net, input)
1-element Array{Float64,1}:
0.8677026536525567
```
In this example, the method is doing the following operations:
```
nnetExecuteLayer(NeuralNetwork, Array{<: Number, 1})
    sigmoid(input * firstHiddenLayer + bias) = sigmoid([1.0] * [1.0] + 1.0) = sigmoid(2) = 0.886
nnetExecuteLayer(NeuralNetwork, Array{<: Number, 1})
    sigmoid(firstHiddenLayer * outputLayer + bias) = sigmoid([0.88] * [1.0] + 1) = sigmoid(1.88) = 0.867
```
"""
function nnetExecute(net::NeuralNetwork, input::Array{<: Number, 1})
    for i in eachindex(net.layers)
        @inbounds input = nnetExecuteLayer(net.layers[i], input, net.biases[i])
    end

    return input
end

"""
    nnetExecute(net::BitNeuralNetwork, input::BitArray{1})

It is the neural network forward

Executes neuron operations through all network layers and returns the
output of the last layer.

Before executing the forward, the method applies a .! operation to all input
elements because in most of cases, 1 is considered as positive and 0 as negative, but 
for xor, the opposite shall be considered. (xnor could be used to it, but it would consume
more CPU depending of the neural network size)

# Example
For this example, consider all the weights are 1 and all biases are 1
```julia_repl
julia> nnetExecute(net, [true])
1-element Array{Float32,1}:
0.0
```
In this example, the method is doing the following operations:
```
nnetExecuteLayer(BitNeuralNetwork, BitArray)
    xor([1], ~[1]) => xor([1], [0]) => count_ones(1) => 1 + 1 => (1 + 1) > (1 - 1) = true
nnetExecuteOutputLayer(BitNeuralNetwork, BitArray)
    xor([1], [1]) => count_ones(0) => 0 + 1 => 1 - (0 + 1) / 1 = 0
```
"""
function nnetExecute(net::BitNeuralNetwork, input::BitArray{1})
    input = .!input

    hiddenLayersNumber = length(net.layers) - 1
    for i in 1:hiddenLayersNumber
        @inbounds input = nnetExecuteLayer(net.layers[i], input, net.biases[i])
    end

    return nnetExecuteOutputLayer(net.layers[length(net.layers)], input, net.biases[length(net.layers)])
end