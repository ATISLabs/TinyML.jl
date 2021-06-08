module DirectConverter
    using Flux
    using ..NEAT
    using ..Misc

    export to_denses

    function copy_weights!(dense::Dense, layer::AbstractArray, indexer::Function)
        for (j, node) in enumerate(layer)
            bias(dense)[j] = node.bias
            for con in node.connections
                weight(dense)[j, indexer(con)] = con.weight
            end
        end
    end

    @inline empty_dense(in, out, σ) = Dense(in,out,σ,
                                        initW=Flux.zeros,
                                        initb=Flux.zeros)

    """
        to_denses(set::NEATCore.TrainingSet{:DFF})

        this function converts NEAT networks to Flux Denses when NEAT's 
        set has feedForward property enabled (:DFF = Deep Feed Forward).
    """
    function to_denses(set::NEAT.NEATCore.TrainingSet{:DFF})
        σ = set.σ
        best = NEAT.NEATCore.getRepresentant!(set)
        layers = best.layers

        denses = Array{Dense, 1}(undef, length(layers))

        in = set.in
        out = length(layers[1])
        denses[1] = empty_dense(set.in, length(layers[1]), σ)

        copy_weights!(denses[1], layers[1], (con) -> con.in)
        for (prev, current) in zip(1:length(denses), 2:length(denses))
            in = out
            out = length(layers[current])
            dense = empty_dense(in, out, σ)
            copy_weights!(dense, layers[current], 
                (con) -> findfirst(x->x.id == con.in, layers[prev]))
            denses[current] = dense
        end

        denses
    end
end