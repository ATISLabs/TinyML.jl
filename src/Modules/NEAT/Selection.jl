function selection!(set::TrainingSet, children::Array{Candidate,1})
    sortSpecies!(set.species)
    killSpeciesExcess(set)
    killBadlyPerformed(set)

    for child in children
        specie = findSpecie(set, child, sorted=true)
        if isnothing(specie)
            specie = Specie()
            push!(set.species, specie)
            push!(specie, child)
        else
            push!(specie, child)
        end
        unsafeAdjustFitness!(specie, child)
    end

    updatepopSize(set)
end

#= Selection functions =#
@inline function sortSpecies!(arr::Array{Specie, 1})
    sort!(arr, by=s->getRepresentant!(s).fitness, rev=true)
end

function killSpeciesExcess(set::TrainingSet)
    if length(set.species) > set.maxSpecies
        set.species = set.species[1:set.maxSpecies]
    end
end

function killBadlyPerformed(set::TrainingSet)
    for specie in set.species
        killingPoint = ceil(Int, length(specie) * set.survivalRate)
        specie.candidates = specie.candidates[1:killingPoint]
    end
end

function δ(set::TrainingSet, cand1::Candidate, cand2::Candidate)
    disjoint, excess, matching, weight = 0,0,0,0
    nFactor = length(cand1.innovations) > length(cand2.innovations) ? 
                length(cand1.innovations) : length(cand2.innovations)
    maxInnov = cand1.maxInnov > cand2.maxInnov ? cand1.maxInnov : cand2.maxInnov
    innovs = union!(collect(keys(cand1.innovations)), collect(keys(cand2.innovations)))

    for innov in innovs
        if haskey(cand1.innovations, innov) && haskey(cand2.innovations, innov)
            matching += 1
            weight += abs(cand1.innovations[innov].weight - cand2.innovations[innov].weight)
        elseif innov < maxInnov
            disjoint += 1
        else
            excess += 1
        end
    end

    return (set.c1*excess + set.c2*disjoint)/nFactor + set.c3*weight/matching
end

@inline sh(set::TrainingSet, delta::Number) = delta < set.deltaThreshold

@inline isSameSpecie(set::TrainingSet, cand1::Candidate, cand2::Candidate) = 
                                                sh(set, δ(set, cand1, cand2))

function findSpecie(set::TrainingSet, cand::Candidate; sorted::Bool=false)
    if sorted
        for specie in set.species
            if isSameSpecie(set, unsafeGetRepresentant(specie), cand)
                return specie
            end
        end
    else
        for specie in set.species
            if isSameSpecie(set, getRepresentant!(specie), cand)
                return specie
            end
        end
    end
    return nothing
end

@inline unsafeAdjustFitness!(specie::Specie, cand::Candidate) = 
            cand.fitness = cand.fitness / length(specie)

@inline adjustFitness!(set::TrainingSet, cand::Candidate) = 
            unsafeAdjustFitness!(findSpecie(set, cand), cand)

function updatepopSize(set::TrainingSet)
    temp = 0
    for specie in set.species
        temp += length(specie)
    end
    set.popSize = temp
end