struct Node <: AbstractNode
    id::UInt32

    output::Ref{Float32}
    b::Float32

    connections::Array{Connection, 1}
    state::Ref{Bool}
end

Node(id::Int) =
    Node(
        id,
        Ref(0.f0),
        rand_bias(),
        [],
        Ref(false)
    )

Node(node::Node; b::Float32) =
    Node(
        id(node),
        output(node),
        b,
        connections(node),
        state(node)
    )

# Getters and Setters
@inline id(node::Node) = node.id

@inline output(node::Node) = node.output[]
@inline output(node::Node, val::Float32) = node.output[] = val

@inline b(node::Node) = node.b
@inline connections(node::Node) = node.connections

@inline state(node::Node) = node.state[]
@inline state!(node::Node) = node.state[] != state(node)