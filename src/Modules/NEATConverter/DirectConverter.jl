module DirectConverter
    using Flux
    using ..NEAT

    export to_denses

    macro copy_weights!(dense, layer, second_indexer)
        quote
            local layer = $(esc(layer))
            local dense = $(esc(dense))
            for (j, node) in enumerate(layer)
                dense.b[j] = node.bias
                for con in node.connections
                    dense.W[j, $(second_indexer)] = con.weight
                end
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

        @copy_weights! denses[1] layers[1] con.in
        for (i, prev, layer) in zip(2:length(layers),
                                    @view(layers[1:end-1]), 
                                    @view(layers[2:end]))
            in = out
            out = length(layer)
            dense = empty_dense(in, out, σ)
            @copy_weights! dense layer findfirst(x->x.id==con.in, :($prev))
            denses[i] = dense
        end

        denses
    end
end