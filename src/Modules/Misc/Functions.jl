@inline printGenAndFitness(gen::Int, fit::AbstractFloat, grant::Bool=true) =
    grant :(println("Gen: $($gen) -- Best Fitness: $($fit)"))