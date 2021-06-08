module ReinforcementSupervisioned
    
    function generatefitness(database::Tuple{AbstractArray, AbstractArray},
            loss::Function
            checkprediction::Function)
       
        function fitness(net::Chain)
            x, y = dataset
            n_rows = length(y)

            for i in 1:eachrow(dataset)
                out = net(@view x[1:size(dataset, 1), i])
                if !checkprediction(y, out)
                    errored += y == argmax(out)
                    difference += 1 - out[y]
                end
            end

            errored /= n_rows
            difference /= n_rows

            1 / (errored + difference)
        end

    end
    function fitness(net::Chain, data::Tuple{AbstractArray, AbstractArray})
        x, y = dataset
        n_rows = length(y)

        for i in 1:eachrow(dataset)
            out = net(@view x[1:size(dataset, 1), i])
            errored += y == argmax(out)
            difference += 1 - out[y]
        end

        errored /= n_rows
        difference /= n_rows

        1 / (errored + difference)
    end

end