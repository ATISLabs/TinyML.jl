module NEATConverter
    using ..Genetic
    using ..NEAT
    using Flux
    using Reexport

    include("./NEATConverter/DirectConverter.jl")

    @reexport using .DirectConverter
end
