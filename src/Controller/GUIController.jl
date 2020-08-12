include("./InputController.jl")
include("./SetController.jl")

function loadJsConstants(d::GUIData)
    w = getWindow(d)
    run(w, string("const GUI_CODE_SEPARATOR = '", GUI_CODE_SEPARATOR, "'"))
    run(w, string("const GUI_CODE_TRAIN = '", GUI_CODE_TRAIN, "'"))
    run(w, string("const GUI_CODE_WATCH = '", GUI_CODE_WATCH, "'"))
    run(w, string("const GUI_CODE_LOAD = '", GUI_CODE_LOAD, "'"))
    run(w, string("const GUI_CODE_SAVE = '", GUI_CODE_SAVE, "'"))
    run(w, string("const GUI_CODE_TRAIN_EXISTING = '", GUI_CODE_TRAIN_EXISTING, "'"))
    run(w, string("const DRAW_MATRIX_SNAKE = '", Snake.DRAW_MATRIX_SNAKE, "'"))
    run(w, string("const DRAW_MATRIX_VOID = '", Snake.DRAW_MATRIX_VOID, "'"))
    run(w, string("const DRAW_MATRIX_FRUIT = '", Snake.DRAW_MATRIX_FRUIT, "'"))

    run(w, "window.requestAnimationFrame(drawGame)")
end

function drawGame(d::GUIData, game::Game)
    w = getWindow(d)
    mat = getDrawingMatrix(game)

    str = "matrix = ["
    limitLine = size(mat)[1] - 1
    limitChar = size(mat)[2] - 1
    for i in 1:limitLine
        str = string(str, "[")
        for j in 1:limitChar
            str = string(str, "'", string(mat[i, j]), "',")
        end
        str = string(str, "'", string(mat[i, size(mat)[2]]), "'],")
    end
    str = string(str, "[")
    for j in 1:limitChar
        str = string(str, "'", string(mat[size(mat)[1], j]), "',")
    end
    str = string(str, "'", string(mat[size(mat)[1], size(mat)[2]]), "']]")

    run(w, str)
end

function runGame(d::GUIData, interval::Float64)
    game = Game()
    move = 0
    while !isLost(game) && move < 50
        drawGame(d, game)
        nextMovement!(getSet(d).chain, game)

        prevSize = length(game.snake.body)
        nextFrame!(game)
        move += 1
        move = (1 - (length(game.snake.body) - prevSize)) * move
        sleep(interval)
    end
end

function trainExisting!(d::GUIData, genCount::Int)
    currentGen = 1
    while currentGen <= genCount
        AI.Train!(getSet(d), 1)
        #guiPrintStats(w, tset, netType, currentGen, genCount)
        runGame(d, GUI_TRAINING_FRAME_INTERVAL)

        currentGen += 1
    end
end

function train!(d::GUIData, args::Array{String, 1})
    nt = getNetType(args)
    input = getSetInput(args)
    setSet!(d, createSet(nt, input...))
    trainExisting!(d, getGenCount(args))
end

function watch(d::GUIData)
    runGame(d, GUI_WATCHING_FRAME_INTERVAL)
end

function loadTrainingSet(d::GUIData)
    if isfile(TSET_NAME)
        setSet!(d, loadSetFromFile())
        setMessage!(d, "Training set loaded")
    else
        setMessage!(d, "No training set file")
    end
end

function saveTrainingSet(d::GUIData)
    saveSetToFile(getSet(d)) 
    setMessage!(d, "Training set saved")
end

function executeAction!(d::GUIData, input::String)
    arr = getInputArray(input)
    op = getCommand(arr)
    args = getArguments(arr)

    w = getWindow(d)

    if op == GUI_CODE_TRAIN
        train!(d, args)
    elseif op == GUI_CODE_LOAD
        loadTrainingSet(d)
    end

    if isSetLoaded(d)
        if op == GUI_CODE_TRAIN_EXISTING
            tset = trainExisting!(d, getGenCount(args))
        elseif op == GUI_CODE_WATCH
            watch(d)
        elseif op == GUI_CODE_SAVE
            saveTrainingSet(d)
        end
    else
        setMessage!(d, "No training set loaded")
    end
end

function startController()
    data = GUIData()
    loadJsConstants(data)
    return data
end