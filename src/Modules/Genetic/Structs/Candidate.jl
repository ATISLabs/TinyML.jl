struct Candidate
    net::Chain
    fitness::Ref{Float32}
end

@inline Candidate(net::Chain) = Candidate(net, Ref(0.f0))

@inline fitness!(cand::Candidate, fitness::AbstractFloat) = cand.fitness[] = Float32(fitness)
@inline fitness(cand::Candidate) = cand.fitness[]
@inline network(cand::Candidate) = cand.net

#= Displays =#
function Base.show(io::IO, t::Candidate)
    print(io, "Candidate(fit=$(fitness(t)))")
end