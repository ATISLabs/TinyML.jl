function mutation!(set::TrainingSet{:NEAT}, children::Array{Candidate,1})
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

function mutation!(set::TrainingSet{:DFF}, children::Array{Candidate,1})
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

@inline mutateAddConnection!(set::TrainingSet, l::Candidate) = addRandomConnection!(set, l)

@inline function mutateAddNode!(set::TrainingSet, l::Candidate)
    conKey = rand(keys(l.connections))
    node = addNode!(set, l)

    addConnection!(set, l, conKey[1], node, con=Connection(l.connections[conKey]))
    addConnection!(set, l, node, conKey[2])
    disableConnection!(l, conKey[1], conKey[2])
end

function mutateAddNode!(set::TrainingSet{:DFF}, cand::Candidate)
    layers = getLayers(cand)
    index = rand(1:length(layers))

    if index == length(layers)
        newLayer = Array{Node, 1}(undef, set.out)
        for i in 1:set.out
            newLayer[i] = getNode(cand, unsafeAddNode!(cand, set.in))
        end
        push!(layers, newLayer)
        cand.outputNodes = newLayer
        for (i, j) in zip(layers[end-1], newLayer)
            addConnection!(set, cand, getId(i), getId(j), con=
                Connection(getId(i), getId(j), w=1))
        end
    else
        nodeId = addNode!(set, cand)
        push!(layers[index], getNode(cand, nodeId))
        addConnection!(set, cand, nodeId, getId(rand(layers[index+1])))    
        if index > 1
            addConnection!(set, cand, getId(rand(layers[index-1])), nodeId)    
        else
            addConnection!(set, cand, rand(1:getIn(set)), nodeId)
        end
    end
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
