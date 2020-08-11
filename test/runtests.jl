using Test

using Flux
include("../src/Modules/BitFlux.jl")
using .BitFlux
include("../src/Modules/Genetic.jl")
using .Genetic
include("../src/Modules/NEAT.jl")
using .NEAT

include("./bitflux.jl")
include("./genetic.jl")
include("./neat.jl")