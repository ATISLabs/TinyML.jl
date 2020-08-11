function crossover!(set::TrainingSet)
    children = Array{Network, 1}(undef, 0)
    #equally divides maximum child number for each species
    maxSpecieChild = floor(Int, (set.maxPopulation - set.populationSize) / length(set.species))
    for specie in set.species
        #it means 'reproductionRate'% of the available number of children will be born
        childLimit = ceil(Int, maxSpecieChild * set.reproductionRate)
        for c in 1:childLimit
            father = specie[rand(1:length(specie))]
            mother = specie[rand(1:length(specie))]

            child = crossoverCandidates(set, father, mother)

            push!(children, child)
        end
    end

    return children
end

#= Crossover functions =#
@inline function getInnovationNumber!(set::TrainingSet)
    set.innovationNumber += 1
    return set.innovationNumber
end

@inline function getInnovation!(set::TrainingSet, in::Int, out::Int)
    if haskey(set.innovations, (in,out))
        return set.innovations[(in,out)]
    else
        innov = getInnovationNumber!(set)
        push!(set.innovations, (in,out) => innov)
        return innov
    end
end

function crossoverCandidates(set::TrainingSet, f::Network, m::Network)
    fittest, other = f.fitness > m.fitness ? (f,m) : (m,f)
    child = deepcopy(fittest)
    for i in length(fittest.nodes)+1:length(other.nodes)
        addNode!(set, child)
    end
    
    if f.fitness == m.fitness
        for (key, con) in other.connections
                if !haskey(fittest.connections, key)
                #possibly add other's nodes
                addConnection!(set, child, key..., con=deepcopy(con))
                else
                child.connections[key].enabled = con.enabled
                child.connections[key].weight = con.weight
                end
        end
    else
        for (key, con) in other.connections
            if haskey(fittest.connections, key)
                if rand(Bool)
                    child.connections[key].enabled = con.enabled
                    child.connections[key].weight = con.weight
                end
            end
        end
    end

    for i in (set.in+set.out+1):(length(child.nodes)+set.in)
        if length(child.nodes[i].connections) == 0
            delete!(child.nodes, i)
        end
    end

    child.fitness = 0
    return child
end
