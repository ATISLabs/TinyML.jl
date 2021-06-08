@inline weight(d::Union{Dense, BitDense}) = d.weight
@inline bias(d::Union{Dense, BitDense}) = d.bias
@inline activation(d::Union{Dense, BitDense}) = d.σ
@inline layers(c::Chain) = c.layers