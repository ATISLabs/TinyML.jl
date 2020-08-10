"""
    abstract type NeuralNetwork

An abstract type for NeuralNetworks

If you want something concrete, try using:
- Float64NeuralNetwork
- BitNeuralNetwork
"""
abstract type NeuralNetwork end


"""
    abstract type MLPNeuralNetwork <: NeuralNetwork

An abstract type for multi layer perceptrons

If you want something concrete, try using:
- Float64NeuralNetwork
- BitNeuralNetwork
"""
abstract type MLPNeuralNetwork <: NeuralNetwork end


"""
    abstract type ConvolutionalNeuralNetwork <: NeuralNetwork

An abstract type for convolutional neural networks

If you want something concrete, try using:
- Float64ConvolutionalNeuralNetwork
- BitConvolutionalNeuralNetwork
"""
abstract type ConvolutionalNeuralNetwork <: NeuralNetwork end


"""
    abstract type CNNLayer

An abstract type for convolutional neural network layers

If you want something concrete, try using:
- ConvolutionalLayer
- MaxPoolLayer
"""
abstract type CNNLayer end


"""
    abstract type ConvLayer end

An abstract type for 4-d layers, such as convolutional layer,
max pool layer.
"""
abstract type ConvLayer <: CNNLayer end


"""
    abstract type FlatLayer

An abstract type for 1-d operations layers, such as flatten layer,
dense layers.
"""
abstract type FlatLayer <: CNNLayer end


"""
    abstract type ConvolutionalLayer <: ConvLayer

An abstract for convolutional layers
"""
abstract type ConvolutionalLayer <: ConvLayer end


"""
    abstract type PoolLayer <: ConvLayer

An abstract type for pooling layers
"""
abstract type PoolLayer <: ConvLayer end


"""
    abstract type FlattenLayer <: FlatLayer

An abstract type for flattening layers.
This type of layer is responsible for reshaping
an input to an 1D array.
"""
abstract type FlattenLayer <: FlatLayer end


"""
    abstract type ActivationLayer <: CNNLayer

An abstract type for activation layers.
This layer type is responsible for applying
an operation to a flattened layer.
"""
abstract type ActivationLayer <: CNNLayer end


"""
    abstract type MLPLayer <: FlatLayer

An abstract type for multi layer perceptron.
"""
abstract type MLPLayer <: FlatLayer end


"""
    abstract type Candidate

Abstract candidate type, it stores all needed data to fitness
function and genetic algorithm

# Note
All childs of Candidate shall have the following fields:
- fitness::Float64
- probability::Float64
- Any type of <:NeuralNetwork
- The needed data for fitness function
"""
abstract type Candidate end