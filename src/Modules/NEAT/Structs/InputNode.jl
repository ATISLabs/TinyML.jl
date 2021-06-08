struct InputNode <: AbstractNode
    id::UInt32
    b::Float32
end

InputNode(id::Int; b::Float32=zero(Float32)) = 
    InputNode(id, b)

InputNode(node::InputNode; b::Float32=b(node)) =
    InputNode(id(node), b)

#Getters and setters
@inline id(node::InputNode) = node.id
@inline b(node::InputNode) = node.b