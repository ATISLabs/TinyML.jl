struct NEATDense <: AbstractNEATLayer
    in::Int
    out::Int
    state::Ref{Bool}

    nodes::Dict{Int, Node}
    output_nodes::Array{Node, 1}

    σ::Function
end

function NEATDense(in::Int, out::Int, σ=identity)
    dense = NEATDense(
        in,
        out,
        false,
        Dict{Int, Node}(),
        [],
        σ
    )
    initialize!(dense, in, out)
    dense
end

function initialize!(layer::AbstractNEATLayer, in::Int, out::Int)
    #input nodes
    for i in 1:in
        push!(layer, InputNode(i))
    end
    #output nodes
    for i in in:(in+out)
        push_output_node!(layer, Node(i))
    end
end

function push_output_node!(layer::AbstractNEATLayer, node::Node)
    push!(output_nodes(layer), node)
    unsafe_push!(layer, node)
end

@inline push!(layer::AbstractNEATLayer, node::Node) =
    push!(nodes(layer), node)

Base.empty!(layer::NEATDense) = empty!(nodes(layer))

# Getters and Setters
in(layer::NEATDense) = layer.in
out(layer::NEATDense) = layer.out
nodes(layer::NEATDense) = layer.nodes
output_nodes(layer::NEATDense) = layer.output_nodes
σ(layer::NEATDense) = layer.σ

state(layer::NEATDense) = layer.state[]
state!(layer::NEATDense) = layer.state[] != state(layer)
