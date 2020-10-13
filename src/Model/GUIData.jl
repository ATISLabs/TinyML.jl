# constants for standardizing data between julia and js
const GUI_CODE_SEPARATOR = ":"
const GUI_CODE_TRAIN = "001"
const GUI_CODE_WATCH = "002"
const GUI_CODE_LOAD = "003"
const GUI_CODE_SAVE = "004"
const GUI_CODE_TRAIN_EXISTING = "005"

#other constants
const GUI_CODE_WAIT_FOR_INPUT = ""

# inverval in seconds between frames
const GUI_TRAINING_FRAME_INTERVAL = 0.01
const GUI_WATCHING_FRAME_INTERVAL = 0.1

mutable struct GUIData
    window::Window

    set::Union{NEAT.TrainingSet, Genetic.TrainingSet}
    setLoaded::Bool

    netType::NetworkType

    function GUIData()
        d = new()

        html = string("file://", joinpath(dirname(dirname(pathof(TinyML))), "assets/index.html"))
        global app = Application()
        w = Window(app, URI(html))

        d.window = w
        d.setLoaded = false

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

@inline getWindow(d::GUIData) = d.window

@inline getNetworkType(d::GUIData) = d.netType
@inline setNetworkType!(d::GUIData, net::AI.NetworkType) = d.netType = net