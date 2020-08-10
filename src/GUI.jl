"""
    guiLoadJsConstants(w::Window)

Passes constant values to js for communication standards.

# Example
```julia_repl
julia> guiLoadJsConstants(w)
```
"""
function guiLoadJsConstants(w::Window)
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

"""
    guiDrawGame(w::Window, game::SnakeGame)

Change js canvas drawing matrix to the current game grid state.

# Example
```julia_repl
julia> guiDrawGame(w, game)
```
NOTE: This function is strictly related to the HTML and js files. Not recommended to use
on your own.
"""
function guiDrawGame(w::Window, game::SnakeGame)
    mat = snakeGetDrawMatrix(game)

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

"""
    guiRunGame(w::Window, candidate::SnakeCandidate, interval::Float64)

Plays a snake game using 'candidate' neural network and draws each frame of the game
in a specified interval.

# Example
```julia_repl
julia> guiRunGame(w, candidate, GUI_WATCHING_FRAME_INTERVAL)
```
NOTE: This function is strictly related to the HTML and js files. Not recommended to use
on your own.
"""
function guiRunGame(w::Window, cand::SnakeCandidate, interval::Float64)
    cand.game = SnakeGame(DEFAULT_MAP_W, DEFAULT_MAP_H, DEFAULT_SNAKE_X,
                        DEFAULT_SNAKE_Y, DEFAULT_DIRECTION)
    move = 0
    while !cand.game.lost && move < 50
        guiDrawGame(w, cand.game)
        aiNextMovement(cand.game, cand.net)

        prevSize = length(cand.game.snake.body)
        snakeNextFrame(cand.game)
        move += 1
        move = (1 - (length(cand.game.snake.body) - prevSize)) * move
        sleep(interval)
    end
end

"""
    guiJsAlert(w::Window, msg::String)

Makes a new js popup with a message written on it.

# Example
```julia_repl
julia> guiJsAlert(w, "Hello World!")
```
"""
guiJsAlert(w::Window, msg::String) = run(w, string("alert('", msg, "')"))

function guiPrintStats(w::Window, tset::TrainingSet, nettype::String, currGen::Int64, genCount::Int64)
    cmd = string("printStats(", 
                    string(currGen), ",",
                    string(tset.candidates[1].fitness), ",",
                    string(tset.popSize), ",",
                    string(genCount), ",",
                    string(tset.selectedCandidatesCount), ",",
                    string(tset.crossoverCuttingFractionDivisor), ",'",
                    nettype, "',",
                    string(tset.mutationRate) ,")")

    run(w, cmd)
end

"""
    guiTrain(w::Window, args)

Generates a new training set and trains it using the specified characteristics of
the GUI form. Also, after each generation it draws the best candidate of the generation
playing a snake game.

# Example
```julia_repl
julia> guiTrain(w, args)
```
NOTE: This function is strictly related to the HTML and js files. Not recommended to use
on your own.
"""
function guiTrain(w::Window, args)
    popSize = parse(Int64, args[1])
    genCount = parse(Int64, args[2])
    selectionNumber = parse(Int64, args[3])
    crossoverDivisor = parse(Int64, args[4])
    mutationiRate = parse(Float64, args[5])

    ccandCode = parse(Int64, args[6])
    createCand = Function
    netType = String
    if ccandCode == 1
        createCand = aiCreateFloat64MLPCandidate
        netType = "Float64 MLP"
    elseif ccandCode == 2
        createCand = aiCreateBitMLPCandidate
        netType = "Binary MLP"
    elseif ccandCode == 3
        createCand = aiCreateFloat64CNNCandidate
        netType = "Float64 CNN"
    else
        createCand = aiCreateBitCNNCandidate
        netType = "Binary CNN"
    end

    currentGen = 1
    tset = aiTrain(createCand, popSize, 1, selectionNumber,
                crossoverDivisor, mutationiRate)
    guiRunGame(w, tset.candidates[1], GUI_TRAINING_FRAME_INTERVAL)
    guiPrintStats(w, tset, netType, currentGen, genCount)

    while currentGen <= genCount
        tset = aiTrain(tset, 1)
        guiPrintStats(w, tset, netType, currentGen, genCount)
        guiRunGame(w, tset.candidates[1], GUI_TRAINING_FRAME_INTERVAL)

        currentGen += 1
    end

    return tset
end

"""
    guiTrainExisting(w::Window, tset, args)

Trains an existing set of candidatess it using the specified characteristics of
the GUI form. Also, after each generation it draws the best candidate of the generation
playing a snake game.

# Example
```julia_repl
julia> guiTrainExisting(w, tset, args)
```
NOTE: This function is strictly related to the HTML and js files. Not recommended to use
on your own.
"""
function guiTrainExisting(w::Window, tset, args)
    if tset != nothing
        genCount = parse(Int64, args[1])
        tset.selectedCandidatesCount = parse(Int64, args[2])
        tset.crossoverCuttingFractionDivisor = parse(Int64, args[3])
        tset.mutationRate = parse(Float64, args[4])
        
        if tset.candidates[1].net isa Float64NeuralNetwork
            netType = "Float64 MLP"
        elseif tset.candidates[1].net isa BitNeuralNetwork
            netType = "Binary MLP"
        elseif tset.candidates[1].net isa Float64ConvolutionalNeuralNetwork
            netType = "Float64 CNN"
        else
            netType = "Binary CNN"
        end
        currentGen = 1

        while currentGen <= genCount
            tset = aiTrain(tset, 1)
            guiPrintStats(w, tset, netType, currentGen, genCount)
            guiRunGame(w, tset.candidates[1], GUI_TRAINING_FRAME_INTERVAL)

            currentGen += 1
        end
    else
        guiJsAlert(w, "No training set loaded")
    end

    return tset
end

"""
    guiWatch(w::Window, tset)

Draws the best candidate of the generation playing a snake game.

# Example
```julia_repl
julia> guiWatch(w, tset)
```
NOTE: This function is strictly related to the HTML and js files. Not recommended to use
on your own.
"""
function guiWatch(w::Window, tset)
    if tset != nothing
        guiRunGame(w, tset.candidates[1], GUI_WATCHING_FRAME_INTERVAL)
    else
        guiJsAlert(w, "No training set loaded")
    end
end

"""
    guiLoadTrainingSet(w::Window)

Loads a training set from file, if it exists.

# Example
```julia_repl
julia> guiLoadTrainingSet(w)
```
NOTE: This function is strictly related to the HTML and js files. Not recommended to use
on your own.
"""
function guiLoadTrainingSet(w::Window)
    if isfile(TSET_NAME)
        tset = aiLoadTrainingSet() 
        guiJsAlert(w, "Training set loaded")
        return tset
    else
        guiJsAlert(w, "No training set to load")
        return nothing
    end
end

"""
    guiSaveTrainingSet(w::Window, tset)

Saves a training set loaded on memory.

# Example
```julia_repl
julia> guiSaveTrainingSet(w, tset)
```
NOTE: This function is strictly related to the HTML and js files. Not recommended to use
on your own.
"""
function guiSaveTrainingSet(w::Window, tset)
    if tset != nothing
        aiSaveTrainingSet(tset) 
        guiJsAlert(w, "Training set saved")
    else
        guiJsAlert(w, "No training set loaded")
    end
end

"""
    guiStartGUI()

Starts and controls the GUI.

# Example
```julia_repl
julia> guiStartGUI()
```
NOTE: This function is strictly related to the HTML and js files. Not recommended to use
on your own.
"""
function guiStartGUI()
    html = string("file://", joinpath(dirname(dirname(pathof(BinarySnake))), "assets/index.html"))

    global app = Application()
    w = Window(app, URI(html))
    guiLoadJsConstants(w)

    ch = msgchannel(w)

    tset = nothing

    try
        while(w.exists)
            msg = split(take!(ch), GUI_CODE_SEPARATOR)

            op = msg[1]
            args = view(msg, 2:length(msg))

            if op == GUI_CODE_TRAIN
                tset = guiTrain(w, args)
            elseif op == GUI_CODE_TRAIN_EXISTING
                tset = guiTrainExisting(w, tset, args)
            elseif op == GUI_CODE_WATCH
                guiWatch(w, tset)
            elseif op == GUI_CODE_LOAD
                tset = guiLoadTrainingSet(w)
            elseif op == GUI_CODE_SAVE
                guiSaveTrainingSet(w, tset)
            end
        end
    catch
        println("Window closed")
    end
end