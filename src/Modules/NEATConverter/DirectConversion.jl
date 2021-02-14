function to_denses(set::NEAT.NEATCore.TrainingSet{:DFF})
    #not used yet
    σ = set.σ
    best = NEAT.NEATCore.getRepresentant!(set)
    layers = best.layers

    denses = Array{Dense, 1}(undef, length(layers))

    in = set.in
    for (i, layer) in enumerate(layers)
        out = length(layer)
        dense = Dense(in, out, initW=Flux.zeros)
        for (j, node) in enumerate(layer)
            for con in node.connections
                if i == 1
                    dense.W[j,con.in] = con.w 
                else
                    dense.W[j,findfirst(x->x.id==con.in, layers[i-1])] = con.w 
                end
            end
            dense.b[j] = node.bias
        end
        denses[i] = dense
        in = out
    end

    #should update chain
    denses
end