module BitFlux
    #= Imports =#
    using Distributions: Uniform
    using Random: bitrand

    #= Constants =#
    const binarynetlib = joinpath(@__DIR__, "BitFlux/ccall/binarynet.so")

    #= Includes =#
    include("./BitFlux/BitTensor.jl")
    include("./BitFlux/BitDense.jl")

    #= Exports =#
    export BitDense, BitTensor
end