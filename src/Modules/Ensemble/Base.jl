module EnsembleCore

    using ..Flux
    using ..BitFlux
    using ..Misc
    using ..Genetic

    include("./Structs/Ensemble.jl")

    export Ensemble, classifiers, combiner

end