include("./SetController.jl")
include("./InputController.jl")

function loadJsConstants(w::Window)
    run(w, string("const GUI_CODE_SEPARATOR = '", GUI_CODE_SEPARATOR, "'"))
    run(w, string("const GUI_CODE_TRAIN = '", GUI_CODE_TRAIN, "'"))
    run(w, string("const GUI_CODE_WATCH = '", GUI_CODE_WATCH, "'"))
    run(w, string("const GUI_CODE_LOAD = '", GUI_CODE_LOAD, "'"))
    run(w, string("const GUI_CODE_SAVE = '", GUI_CODE_SAVE, "'"))
    run(w, string("const GUI_CODE_TRAIN_EXISTING = '", GUI_CODE_TRAIN_EXISTING, "'"))
    run(w, string("const DRAW_MATRIX_SNAKE = '", DRAW_MATRIX_SNAKE, "'"))
    run(w, string("const DRAW_MATRIX_VOID = '", DRAW_MATRIX_VOID, "'"))
    run(w, string("const DRAW_MATRIX_FRUIT = '", DRAW_MATRIX_FRUIT, "'"))

    run(w, "window.requestAnimationFrame(drawGame)")
end

function drawGame(w::Window, game::Game)
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
        drawGame(w, game)
        nextMovement!(cand, game)

        prevSize = length(game.snake.body)
        nextFrame!(game)
        move += 1
        move = (1 - (length(game.snake.body) - prevSize)) * move
        sleep(interval)
    end
end

function trainExisting!(d::GUIData, args...)
    currentGen = 1
    while currentGen <= getGenCount(args)
        AI.Train!(getSet(d), 1)
        #guiPrintStats(w, tset, netType, currentGen, genCount)
        runGame(d)

        currentGen += 1
    end

    return tset
end

function Train!(d::GUIData, args::Array{String, 1})
    nt = getNetType(args)
    input = getTrainingInput(args)
    setSet!(d, createSet(nt, args...))
    trainExisting!(d, args)
    return set
end

function watch(d::GUIData)
    runGame(d, GUI_WATCHING_FRAME_INTERVAL)
end

function loadTrainingSet(d::GUIData)
    if isfile(TSET_NAME)
        setSet!(d, loadSetFromFile())
    else
        setMessage!(d, "No training set file")
    end
end

function executeAction!(d::GUIData, input::String)
    arr = getArgumentArray(input)
    op = getCommand(input)
    args = getInputArguments(input)

    w = getWindow(d)

    if op == GUI_CODE_TRAIN
        setSet!(d, Train!(w, args))
    elseif op == GUI_CODE_LOAD
        loadTrainingSet!(d)
    end

    if isSetLoaded(d)
        if op == GUI_CODE_TRAIN_EXISTING
            tset = guiTrainExisting(w, tset, args)
        elseif op == GUI_CODE_WATCH
            watch(d)
        elseif op == GUI_CODE_SAVE
            guiSaveTrainingSet(w, tset)
        end
    else
        setMessage!(d, "No training set loaded")
    end
end

function startController()
    data = GUIData()
    loadJsConstants(getWindow(d))
    return data
end