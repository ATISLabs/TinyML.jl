function mutation!(set::TrainingSet, children::Array{Network,1})
    for child in children
        if set.addConnectionMutationRate > rand()
            mutateAddConnection!(set, child)
        end
        if set.addNodeMutationRate > rand()
            mutateAddNode!(set, child)
        end
        for (key, con) in child.connections
            if set.weightMutationRate > rand()
                mutateWeight!(con)
            end
            if set.toggleConnectionMutationRate > rand()
                mutateConnection!(con)
            end
        end
        for (key, node) in child.nodes
            if set.biasMutationRate > rand()
                mutateBias!(node)
            end
        end
    end
end

#= Mutation functions =#
@inline function mutateBias!(node::Node)
    node.bias = randBias()
end

@inline mutateAddConnection!(set::TrainingSet, l::Network) = addRandomConnection!(set, l)

@inline function mutateAddNode!(set::TrainingSet, l::Network)
    conKey = rand(keys(l.connections))
    node = addNode!(set, l)

    addConnection!(set, l, conKey[1], node, con=Connection(l.connections[conKey]))
    addConnection!(set, l, node, conKey[2])
    disableConnection!(l, conKey[1], conKey[2])
end

@inline function mutateWeight!(c::Connection)
    c.weight = randWeight()
end

@inline function mutateConnection!(c::Connection)
    c.enabled = !c.enabled
end

@inline function disableConnection!(l::NEATLayer, in::Int, out::Int)
    l.connections[(in, out)].enabled = false
end
