module TinyML
    using Reexport

    include("./Modules/BitFlux.jl")
    include("./Modules/Genetic.jl")
    include("./Modules/NEAT.jl")
    
    @reexport using .BitFlux
    using .Genetic
    using .NEAT

    export Genetic, NEAT
end
