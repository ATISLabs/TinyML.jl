module NEATConverter
    using ..Genetic
    using ..NEAT
    using ..Misc
    using Flux
    using Reexport

    include("./NEATConverter/DirectConverter.jl")

    @reexport using .DirectConverter
end
