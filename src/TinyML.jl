module TinyML
    using Reexport

    include("./Modules/BitFlux.jl")
    include("./Modules/Genetic.jl")
    include("./Modules/NEAT.jl")
    include("./Modules/NEATConverter.jl")
    include("./Modules/Ensemble.jl")
    
    @reexport using .BitFlux
    using .Genetic
    using .NEAT
    using .NEATConverter
    using .Ensemble

    export Genetic, NEAT, NEATConverter, Ensemble
end
