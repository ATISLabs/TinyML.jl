# Snake.jl
    # Constants related to Snake's directions
    const UP = 1
    const DOWN = 2
    const RIGHT = 3
    const LEFT = 4

    # Constants related to drawing matrix
    const DRAW_MATRIX_SNAKE = 'X'
    const DRAW_MATRIX_VOID = ' '
    const DRAW_MATRIX_FRUIT = 'O'

    # Array of constant positions to be used until Snake reaches the length of 10
    const FOODPOS = [[10,10], [5, 5], [14, 14], [7, 5], [13, 4], [4, 13]]

    # Constant related to Snake's length limit until it randomizes Snake position
    const SNAKE_MAX_SIZE_TO_RANDOM = 10


# AI.jl
    const TSET_NAME = "tset.jld2"
    const AI_MAX_MOVEMENT = 200
    const sensorCount = 20
    

# GUI.jl
    # constants for standardizing data between julia and js
    const GUI_CODE_SEPARATOR = ":"
    const GUI_CODE_TRAIN = "001"
    const GUI_CODE_WATCH = "002"
    const GUI_CODE_LOAD = "003"
    const GUI_CODE_SAVE = "004"
    const GUI_CODE_TRAIN_EXISTING = "005"

    # inverval in seconds between frames
    const GUI_TRAINING_FRAME_INTERVAL = 0.01
    const GUI_WATCHING_FRAME_INTERVAL = 0.1