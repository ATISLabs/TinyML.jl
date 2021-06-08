module BitFlux
    #= Imports =#
    using Distributions: Uniform
    using Random: bitrand

    #= Exports =#
    export BitDense, BitTensor, chunks

    #= Includes =#
    include("./BitFlux/BitTensor.jl")
    include("./BitFlux/BitDense.jl")
end