module Sampler

    using ..EnsembleCore

    weightsample(array::AbstractVector) =
        Weights(ones(Float32, length(array)))

end