
jsAlert(w::Window, msg::String) = run(w, string("alert('", msg, "')"))


function guiSaveTrainingSet(w::Window, tset::Union{NEAT.TrainingSet,Genetic.TrainingSet})
    if !isnothing(tset)
        aiSaveTrainingSet(tset) 
        guiJsAlert(w, "Training set saved")
    else
        guiJsAlert(w, "No training set loaded")
    end
end

function startGUI()

    d = startController()
    #try
        while(getWindow(d).exists)
            msg = take!(msgchannel(getWindow(d)))
            executeAction!(d, msg)
            if hasMessage(d)
                jsAlert(getWindow(d), getMessage(d))
        end
    #=catch
        println("Window closed")
    end=#
end