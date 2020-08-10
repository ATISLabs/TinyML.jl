"""
    mutable struct SnakeCandidate <: Candidate

Extends of Candidate and stores data for genetic algorithm

# Structure
```julia
game::SnakeGame
nn::NeuralNetwork
fitness::Float64
probability::Float64
```
 - game: Stores game data
 - nn: Neural network, it can be any type extended of NeuralNetwork
 - fitness: Stores the fitness value of some evaluation
 - probability: Stores the probability value for roulette selection (needed only if using genSelectionRoulette())

 # Initialization
 ```julia_repl
julia> SnakeCandidate(game::SnakeGame, nn::NeuralNetwork)
 ```

 # Example
 ```julia_repl
julia> SnakeCandidate(SnakeGame(10, 10, 5, 5, UP), BitNeuralNetwork(20, [20, 20], 4))
 ```
"""
mutable struct SnakeCandidate <: Candidate
    game::SnakeGame
    net::NeuralNetwork
    fitness::Float64
    probability::Float64

    function SnakeCandidate(game::SnakeGame, net::NeuralNetwork)
        cand = new()

        cand.game = game
        cand.net = net
        cand.fitness = 0
        cand.probability = 0

        return cand
    end
end

"""
    aiGetFloat64SensorsData(game::SnakeGame)

Returns a 20-length float64 array to be used as input for neural network.
It contains the following values, where 1 is true and 0 is false:
 - 4 sensors checking if the fruit is on one of the snake head sides pointing a line
 - 4 sensors checking if the fruit is on one of the snake sides
 - 4 sensors checking if a piece of the snake body is on one of the snake head sides pointing a line
 - 4 sensors checking if a piece of the snake body is on one of the snake sides
 - 4 sensors checking if the snake head is near to the wall

 # Example
 ```julia_repl
 julia> aiGetFloat64SensorsData(game)
 20-element Array{Float64,1}:
 0.0
 0.0
 1.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 0.0
 1.0
 0.0
 0.0
 0.0
 1.0
 0.0
 0.0
 0.0
 0.0
 0.0
 ```
"""
function aiGetFloat64SensorsData(game::SnakeGame)
    data = Array{Float64, 1}(undef, 0)

    #fruit is on the left side
    onLeft = game.snake.head.x > game.fruit.x
    #fruit is on the right side
    onRight = game.snake.head.x < game.fruit.x
    #fruit is on the middle
    onMiddle = game.snake.head.x == game.fruit.x

    #FRUIT DIRECTION
    #fruit is at snake's left
    push!(data, onLeft && game.snake.head.y == game.fruit.y)
    #fruit is at snake's right
    push!(data, onRight && game.snake.head.y == game.fruit.y)
    #fruit is at snake's top
    push!(data, onMiddle && game.snake.head.y > game.fruit.y)
    #fruit is at snake's down
    push!(data, onMiddle && game.snake.head.y < game.fruit.y)

    #FRUIT SIDES
    #fruit is on left
    push!(data, onLeft)
    #fruit is on right
    push!(data, onRight)
    #fruit is on top
    push!(data, game.snake.head.y > game.fruit.y)
    #fruit is on bottom
    push!(data, game.snake.head.y < game.fruit.y)

    #SNAKE BODY
    #BODY SIDES
    bodySides = falses(4)
    bodyQuadrants = falses(4)
    for i in 2:length(game.snake.body)
        xEqual::Bool = game.snake.head.x == game.snake.body[i].x
        yEqual::Bool = game.snake.head.y == game.snake.body[i].y

        #a piece of the snake body is at snake's left
        bodySides[1] = bodySides[1] || (yEqual && game.snake.head.x > game.snake.body[i].x)
        #a piece of the snake body is at snake's right
        bodySides[2] = bodySides[2] || (yEqual && game.snake.head.x < game.snake.body[i].x)
        #a piece of the snake body is at snake's top
        bodySides[3] = bodySides[3] || (xEqual && game.snake.head.y > game.snake.body[i].y)
        #a piece of the snake body is at snake's bottom
        bodySides[4] = bodySides[4] || (xEqual && game.snake.head.y < game.snake.body[i].y)

        yOver = game.snake.head.y > game.snake.body[i].y
        yUnder = game.snake.head.y < game.snake.body[i].y
        #a piece of the snake is on the right
        bodyQuadrants[1] = bodyQuadrants[1] || game.snake.head.x < game.snake.body[i].x
        #a piece of the snake is on the left
        bodyQuadrants[2] = bodyQuadrants[2] || game.snake.head.x > game.snake.body[i].x
        #a piece of the snake is on the top
        bodyQuadrants[3] = bodyQuadrants[3] || yOver
        #a piece of the snake is on the bottom
        bodyQuadrants[4] = bodyQuadrants[4] || yUnder
    end
    append!(data, bodySides)
    append!(data, bodyQuadrants)

    #NEAR TO WALL
    #top wall
    push!(data, game.snake.head.y == 1)
    #left wall
    push!(data, game.snake.head.x == 1)
    #bottom wall
    push!(data, game.snake.head.y == game.mapH)
    #right wall
    push!(data, game.snake.head.x == game.mapW)

    return data
end

"""
    aiGetBitSensorsData(game::SnakeGame)

Returns a 20-length bit array to be used as input for neural network.
It contains the following values, where 1 is true and 0 is false:
 - 4 sensors checking if the fruit is on one of the snake head sides pointing a line
 - 4 sensors checking if the fruit is on one of the snake sides
 - 4 sensors checking if a piece of the snake body is on one of the snake head sides pointing a line
 - 4 sensors checking if a piece of the snake body is on one of the snake sides
 - 4 sensors checking if the snake head is near to the wall

 # Example
 ```julia_repl
 julia> aiGetFloat64SensorsData(game)
 20-element BitArray{1}:
 0
 0
 1
 0
 0
 0
 0
 0
 0
 0
 1
 0
 0
 0
 1
 0
 0
 0
 0
 0
 ```
"""
function aiGetBitSensorsData(game::SnakeGame)
    data = BitArray{1}(undef, 0)

    #fruit is on the left side
    onLeft = game.snake.head.x > game.fruit.x
    #fruit is on the right side
    onRight = game.snake.head.x < game.fruit.x
    #fruit is on the middle
    onMiddle = game.snake.head.x == game.fruit.x

    #FRUIT DIRECTION
    #fruit is at snake's left
    push!(data, onLeft && game.snake.head.y == game.fruit.y)
    #fruit is at snake's right
    push!(data, onRight && game.snake.head.y == game.fruit.y)
    #fruit is at snake's top
    push!(data, onMiddle && game.snake.head.y > game.fruit.y)
    #fruit is at snake's down
    push!(data, onMiddle && game.snake.head.y < game.fruit.y)

    #FRUIT SIDES
    #fruit is on left
    push!(data, onLeft)
    #fruit is on right
    push!(data, onRight)
    #fruit is on top
    push!(data, game.snake.head.y > game.fruit.y)
    #fruit is on bottom
    push!(data, game.snake.head.y < game.fruit.y)

    #SNAKE BODY
    #BODY SIDES
    bodySides = falses(4)
    bodyQuadrants = falses(4)
    for i in 2:length(game.snake.body)
        xEqual::Bool = game.snake.head.x == game.snake.body[i].x
        yEqual::Bool = game.snake.head.y == game.snake.body[i].y

        #a piece of the snake body is at snake's left
        bodySides[1] = bodySides[1] || (yEqual && game.snake.head.x > game.snake.body[i].x)
        #a piece of the snake body is at snake's right
        bodySides[2] = bodySides[2] || (yEqual && game.snake.head.x < game.snake.body[i].x)
        #a piece of the snake body is at snake's top
        bodySides[3] = bodySides[3] || (xEqual && game.snake.head.y > game.snake.body[i].y)
        #a piece of the snake body is at snake's bottom
        bodySides[4] = bodySides[4] || (xEqual && game.snake.head.y < game.snake.body[i].y)

        yOver = game.snake.head.y > game.snake.body[i].y
        yUnder = game.snake.head.y < game.snake.body[i].y
        #a piece of the snake is on the right
        bodyQuadrants[1] = bodyQuadrants[1] || game.snake.head.x < game.snake.body[i].x
        #a piece of the snake is on the left
        bodyQuadrants[2] = bodyQuadrants[2] || game.snake.head.x > game.snake.body[i].x
        #a piece of the snake is on the top
        bodyQuadrants[3] = bodyQuadrants[3] || yOver
        #a piece of the snake is on the bottom
        bodyQuadrants[4] = bodyQuadrants[4] || yUnder
    end
    append!(data, bodySides)
    append!(data, bodyQuadrants)

    #NEAR TO WALL
    #top wall
    push!(data, game.snake.head.y == 1)
    #left wall
    push!(data, game.snake.head.x == 1)
    #bottom wall
    push!(data, game.snake.head.y == game.mapH)
    #right wall
    push!(data, game.snake.head.x == game.mapW)

    return data
end

"""
    aiCreateFloat64MLPCandidate()

Generates a SnakeCandidate containing a Float64 MLP predefined for using on
genetic algorithm.

# Example
```julia_repl
julia> aiCreateFloat64MLPCandidate()
```
"""
aiCreateFloat64MLPCandidate() = SnakeCandidate(SnakeGame(DEFAULT_MAP_W, DEFAULT_MAP_H, DEFAULT_SNAKE_X,
                                        DEFAULT_SNAKE_Y, DEFAULT_DIRECTION), 
                                     Float64NeuralNetwork(20, [20, 20], 4))

"""
    aiCreateBitMLPCandidate()

Generates a SnakeCandidate containing a Bit MLP predefined for using on
genetic algorithm.

# Example
```julia_repl
julia> aiCreateBitMLPCandidate()
```
"""
aiCreateBitMLPCandidate() = SnakeCandidate(SnakeGame(DEFAULT_MAP_W, DEFAULT_MAP_H, DEFAULT_SNAKE_X,
                                        DEFAULT_SNAKE_Y, DEFAULT_DIRECTION),
                                        BitNeuralNetwork(20, [20, 20], 4))

"""
    aiCreateFloat64CNNCandidate()

Generates a SnakeCandidate containing a Float64 CNN predefined for using on
genetic algorithm.

# Example
```julia_repl
julia> aiCreateFloat64CNNCandidate()
```
"""

"""
    aiNextMovement(game::SnakeGame, net::Float64NeuralNetwork)

Changes snake direction based on the output of a neural network.
To convert neural network output to direction, it gets the index of the highest value on output
array, which is interpreted as a direction.

# Example
```julia_repl
julia> aiNextMovement(game, net)
```
"""
function aiNextMovement(game::SnakeGame, net::Float64NeuralNetwork)
        snakeSetDirection(game, findmax(nnetExecute(net, aiGetFloat64SensorsData(game)))[2])
end

"""
    aiNextMovement(game::SnakeGame, net::BitNeuralNetwork)

Changes snake direction based on the output of a neural network.
To convert neural network output to direction, it gets the index of the highest value on output
array, which is interpreted as a direction.

# Example
```julia_repl
julia> aiNextMovement(game, net)
```
"""
function aiNextMovement(game::SnakeGame, net::BitNeuralNetwork)
        snakeSetDirection(game, findmax(nnetExecute(net, aiGetBitSensorsData(game)))[2])
end

"""
    aiCalculateFruitDistance(game::SnakeGame)

Calculates the euclidean distance between the fruit and the snake head

# Example
```julia_repl
julia> aiNextMovement(game)
2
```
"""
function aiCalculateFruitDistance(game::SnakeGame)
    return sqrt((game.snake.head.x - game.fruit.x)^2 + (game.snake.head.y - game.fruit.y)^2)
end

"""
    aiFitnessEvaluate(candidate::SnakeCandidate)

Evaluates a candidate by playing the snake game and sets its fitness value. 
Also, it returns the candidate passed to the function as an argument.

# Example
```julia_repl
julia> aiFitnessEvaluate(candidate)
```
Execution:
- Define fitness and probability to 0 and start a new game
- Loop until the game is lost
    - Calculate the distance between the snake and the fruit
    - If that distance is smaller than on the previous frame, do
        - Add '1/(distance + movimentCount)' to fitness
    - else
        - Subtract '1 / (distance + movimentCount)' to fitness
- Add '5 * snakeLength' to fitness

By doing the steps above, it is possible to estimate if a candidate
is playing well or not by adding points if the snake eats the fruit and
if it goes near to a fruit.
"""
function aiFitnessEvaluate(cand::SnakeCandidate)
    cand.fitness = 0

    maxMovement = AI_MAX_MOVEMENT
    movCount = 0

    cand.game = SnakeGame(DEFAULT_MAP_W, DEFAULT_MAP_H, DEFAULT_SNAKE_X, 
        DEFAULT_SNAKE_Y, DEFAULT_DIRECTION)

    while !cand.game.lost && movCount < AI_MAX_MOVEMENT &&
            length(cand.game.snake.body) < SNAKE_MAX_SIZE_TO_RANDOM
        aiNextMovement(cand.game, cand.net)
        prevSize = length(cand.game.snake.body)

        distanceA = aiCalculateFruitDistance(cand.game)

        snakeNextFrame(cand.game)

        distanceB = aiCalculateFruitDistance(cand.game)
        movCount += 1
        
        if distanceB < distanceA
            cand.fitness += 1 / (distanceB + movCount)
            distanceA = distanceB
        else
            cand.fitness -= 1 / (distanceB + movCount)
        end

        movCount = (1 - (length(cand.game.snake.body) - prevSize)) * movCount
    end
    cand.fitness += 5 * (length(cand.game.snake.body) - 1) + 1

    return cand
end

"""
    aiSaveTrainingSet(tset::TrainingSet)

Saves the content of a training set variable named 'tset.jld2'.

# Example
```julia_repl
julia> aiSaveTrainingSet(tset)
```
"""
aiSaveTrainingSet(arr::TrainingSet) = @save TSET_NAME arr

"""
    aiLoadTrainingSet()

Loads the content of a saved training set

# Example
```julia_repl
julia> candidates = aiLoadTrainingSet()
```
"""
function aiLoadTrainingSet()
    arr = :TrainingSet

    @load TSET_NAME arr
    
    return arr
end

"""
    aiDeleteTrainingSet()

Deletes a saved training.

# Example
```julia_repl
julia> aiDeleteTrainingSet()
```
"""
aiDeleteTrainingSet() = rm(TSET_NAME)

function aiTrain(tset::TrainingSet, genCount::Integer)
    return genExecute(tset, genCount)
end

"""
    aiTrain(createCandidate::Function, populationSize::Int, genCount::Int, selectedCandidatesCount::Int,
                crossoverDivisorFactor::Int, mutationRate::Float64)

Executes genetic algorithm and passes all necessary data for snake game ai. It returns a TrainingSet variable
trained after the specified number of generations.

# Example
```julia_repl
julia> aiTrain(aiCreateBitMLPCandidate, 2000, 100, 10, 2, 0.1)
```
"""
function aiTrain(createCand::Function, popSize::Integer, genCount::Integer, selectedCandidatesCount::Integer,
    crossoverDivisorFactor::Integer, mutationRate::Float64)
    return genExecute(createCand, aiFitnessEvaluate, popSize, genCount, 
            selectedCandidatesCount, crossoverDivisorFactor, mutationRate)
end