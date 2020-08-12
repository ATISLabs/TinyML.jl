mutable struct GUIData
    window::Window

    set::Union{NEAT.TrainingSet, Genetic.TrainingSet}
    setLoaded::Bool

    message::String
    isRead::Bool

    function GUIData()
        d = new()

        html = string("file://", joinpath(dirname(dirname(pathof(BinarySnake))), "assets/index.html"))
        global app = Application()
        w = Window(app, URI(html))

        d.window = w
        d.setLoaded = false
        d.isRead = true
        d.message = ""

        return d
    end
end

@inline isSetLoaded(d::GUIData) = d.setLoaded
@inline setSetLoaded!(d::GUIData, value::Bool) = d.setLoaded = value

function setSet!(d::GUIData, set::Union{NEAT.TrainingSet, Genetic.TrainingSet})
    d.set = set
    setSetLoaded!(d, true)
end
@inline getSet(d::GUIData) = isSetLoaded(d) ? d.set : nothing

@inline setRead!(d::GUIData, value::Bool) = d.isRead = value
@inline hasMessage(d::GUIData) = !d.isRead

function setMessage!(d::GUIData, msg::String)
    setRead!(d, false)
    d.message = msg
end
function getMessage(d::GUIData)
    setRead!(d, true)
    return d.message
end

@inline getWindow(d::GUIData) = d.window