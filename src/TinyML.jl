module TinyML
    using Reexport

    include("./Modules/BitFlux.jl")
    include("./Modules/Misc.jl")
    include("./Modules/Genetic.jl")
    include("./Modules/NEAT.jl")
    include("./Modules/NEATConverter.jl")
    include("./Modules/Ensembles.jl")
    
    @reexport using .BitFlux
    using .Misc
    using .Genetic
    using .NEAT
    using .NEATConverter
    @reexport using .Ensembles

    export Genetic, NEAT, NEATConverter, Ensembles
end
