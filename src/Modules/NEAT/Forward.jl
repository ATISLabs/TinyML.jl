function (l::NEATDense)(input::Array{<:Number,1})
    forward(l, input, l.σ)
end

function (n::Candidate)(set::TrainingSet, input::Array{<:Number,1})
    forward(n, input, set.σ)
end

@inline (e::EvaluationLayer)(input) = forward(e.net, input, e.σ)

@inline (e::NEATDense)(input) = forward(e, input, e.σ)

function forward(l::NEATLayer, input::Array{<:Number,1}, σ::Function)
    l.outputState = !l.outputState
    return [getNodeOutput(node, input, length(input)+1, l.nodes, l.outputState, σ) for node in l.outputNodes]
end

function getNodeOutput(node::Node, input::Array{<:Number,1}, inputLim::Int, 
            nodes::Dict{Int,Node}, currState::Bool, σ::Function)
    if node.outputState != currState
        temp = node.bias
        #For cyclic graphs
        node.output = 0
        node.outputState = currState
        for con in node.connections
            #nodes with ids lesser or equal to input size are sensor nodes
            if con.in < inputLim
                temp += con.weight * con.enabled * input[con.in]
            #if the node isn't a sensor node, it's a normal nodes
            elseif nodes[con.in].outputState == currState
                temp += con.weight * con.enabled * nodes[con.in].output
            #if the node isn't calculated at this point, create a recursion
            else
                temp += con.weight * con.enabled * getNodeOutput(nodes[con.in], 
                            input, inputLim, nodes, currState, σ)
            end
        end
        node.output = σ(temp)
    end

    return node.output
end
