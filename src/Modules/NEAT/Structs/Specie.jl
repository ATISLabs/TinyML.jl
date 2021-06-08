const Specie = Array{Candidate, 1}

#Other
Base.sort!(s::Specie) = sort!(s.candidates, by=c->c.fitness, rev=true)

#Displays
Base.show(io::IO, s::Specie) =
    print(io, "Specie($(length(s.candidates)))")