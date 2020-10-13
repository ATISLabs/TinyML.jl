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


function runGame(d::GUIData, interval::Float64)
    game = Game()
    moveCount = 0
    while !isLost(game) && moveCount < 50
        drawGame(d, game)
        nextMovement!(getSet(d).chain, game)

        prevSize = length(game.snake.body)
        nextFrame!(game)
        moveCount += 1
        moveCount = (1 - (length(game.snake.body) - prevSize)) * moveCount
        sleep(interval)
    end
end

function trainExisting!(d::GUIData, genCount::Int)
    max = 6
    if genCount < max
        max = genCount
    end
    currentGen = 0

    for i in 0:max
        interval = floor(Int, i * genCount / max) - currentGen
        currentGen += interval
        printStats(d, currentGen, genCount, 
                    AI.snakeFitness(getSet(d).chain),
                    getSet(d).popSize, getNetworkType(d))
        runGame(d, GUI_TRAINING_FRAME_INTERVAL)
        AI.train!(getSet(d), interval)
    end
end

function train!(d::GUIData, args::Array{String, 1})
    nt = getNetType(args)
    input = getSetInput(args)
    setNetworkType!(d, nt)
    setSet!(d, createSet(nt, input...))
    trainExisting!(d, getGenCount(args))
end

function watch(d::GUIData)
    runGame(d, GUI_WATCHING_FRAME_INTERVAL)
end

function loadTrainingSet(d::GUIData)
    if isfile(TSET_NAME)
        setSet!(d, loadSetFromFile())
        jsAlert(d, "Training set loaded")
    else
        jsAlert(d, "No training set file")
    end
end

function saveTrainingSet(d::GUIData)
    saveSetToFile(getSet(d)) 
    jsAlert(d, "Training set saved")
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
        jsAlert(d, "No training set loaded")
    end
end

function startController()
    data = GUIData()
    loadJsConstants(data)
    return data
end

function startGUI()
    d = startController()
    #try
        while(getWindow(d).exists)
            executeAction!(d, getInput(d))
        end
    #=catch
        println("Window closed")
    end=#
end