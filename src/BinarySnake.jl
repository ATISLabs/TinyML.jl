module BinarySnake

const PKGNAME = "BinarySnake"

using LinearAlgebra
using Electron
using JLD2
using Pkg
using Distributions

include("./AbstractTypes.jl")
include("./Constants.jl")
include("./BinMatrix.jl")
include("./Snake.jl")
include("./Activation.jl")
include("./NeuralNetwork.jl")
include("./Genetic.jl")
include("./AI.jl")
include("./GUI.jl")
include("./Exports.jl")

export startBinarySnake

const DEFAULT_MAP_W = 15
const DEFAULT_MAP_H = 15
const DEFAULT_SNAKE_X = 8
const DEFAULT_SNAKE_Y = 8
const DEFAULT_DIRECTION = UP

"""
    startBinarySnake()

Don't forget to bring a towel!
"""
function startBinarySnake()
    guiStartGUI()
end

end