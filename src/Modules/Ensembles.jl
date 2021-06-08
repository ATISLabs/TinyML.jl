module Ensembles

    using Base: AbstractVecOrTuple
    using Flux
    using Reexport
    using ..BitFlux
    using ..Misc
    using ..Genetic
    
    include("./Ensemble/Base.jl")
    include("./Ensemble/Sampler.jl")
    #include("./Ensemble/Bagging.jl")
    include("./Ensemble/Boosting.jl")
   
    @reexport using .EnsembleCore
    @reexport using .Sampler
    @reexport using .Boosting

end