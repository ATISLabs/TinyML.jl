module BitFlux
    #= Imports =#
    using Distributions: Uniform
    using Random: bitrand

    #= Exports =#
    export BitDense, BitTensor

    #= Constants =#
    const binarynetlib = joinpath(@__DIR__, "BitFlux/ccall/binarynet.so")

    #= Includes =#
    include("./BitFlux/BitTensor.jl")
    include("./BitFlux/BitDense.jl")
end