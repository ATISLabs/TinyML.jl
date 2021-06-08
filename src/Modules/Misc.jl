module Misc

    using Flux
    using ..BitFlux

    include("./Misc/Functions.jl")
    include("./Misc/Flux.jl")

    export weight, bias, layers, activation

end