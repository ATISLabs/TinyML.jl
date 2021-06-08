using Test

using Flux
using TinyML

using Random
Random.seed!(10)

include("./bitflux.jl")
include("./genetic.jl")
include("./neat.jl")
include("./converter.jl")
#include("./ensemble.jl")