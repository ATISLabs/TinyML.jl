module TinyML
    using Reexport

    include("./Modules/BitFlux.jl")
    include("./Modules/Genetic.jl")
    include("./Modules/NEAT.jl")
    include("./Modules/NEATConverter.jl")
    
    @reexport using .BitFlux
    using .Genetic
    using .NEAT
    using .NEATConverter

    export Genetic, NEAT, NEATConverter
end
