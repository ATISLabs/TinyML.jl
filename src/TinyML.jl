module TinyML
    include("./Modules/BitFlux.jl")
    include("./Modules/Genetic.jl")
    include("./Modules/NEAT.jl")
    include("./Modules/Snake.jl")
    include("./Controller/AI.jl")
    
    using .BitFlux
    using .Genetic
    using .NEAT
    using .Snake
    using .AI

    using Electron
    using JLD2

    include("./Model/GUIData.jl")
    include("./View/GUIView.jl")
    include("./Controller/GUIController.jl")

    startTinyML() = startGUI()

    export startTinyML
    
end
