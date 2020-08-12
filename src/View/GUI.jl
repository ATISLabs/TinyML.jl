jsAlert(w::Window, msg::String) = run(w, string("alert('", msg, "')"))

function startGUI()
    d = startController()
    #try
        while(getWindow(d).exists)
            msg = take!(msgchannel(getWindow(d)))
            executeAction!(d, msg)
            if hasMessage(d)
                jsAlert(getWindow(d), getMessage(d))
            end
        end
    #=catch
        println("Window closed")
    end=#
end