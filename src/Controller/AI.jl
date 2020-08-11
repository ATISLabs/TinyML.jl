module AI
    using ..Snake

    export Train!, createBitGeneticSet, createFloatGeneticSet, 
            createFloatNEATSet, createGeneticSet
    export NetworkType, FloatMLP, BitMLP, FloatNEAT, BitNEAT, FloatCNN, BitCNN

    const AI_MAX_MOVEMENT = 200
    const sensorCount = 20

    @enum NetworkType FloatMLP BitMLP FloatNEAT BitNEAT FloatCNN BitCNN

    const inputFunction = Ref{Function}(identity)

    function getFloatSensorsData(game::SnakeGame)
        data = Array{Float32, 1}(undef, 0)

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
        push!(data, game.snake.head.y == MAP_SIZE)
        #right wall
        push!(data, game.snake.head.x == MAP_SIZE)

        return data
    end

    function getBitSensorsData(game::SnakeGame)
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
        push!(data, game.snake.head.y == MAP_SIZE)
        #right wall
        push!(data, game.snake.head.x == MAP_SIZE)

        return data
    end

    function setInputFunction!(x::NetworkType)
        f = nothing
        if x == FloatMLP
            f = getFloatSensorsData
        elseif x == BitMLP
            f = getBitSensorsData
        elseif x == FloatNEAT
            f = getFloatSensorsData
        elseif x == BitNEAT
            f == getBitSensorsData
        elseif x == FloatCNN
            f = getFloatDrawingMatrix
        else
            #f = aiGetBitDrawingMatrix
        end
        inputFunction[] = f
    end

    @inline getInputFunction() = inputFunction[]

    function nextMovement!(net::Chain, game::Game)
        setDirection!(game, findmax(net(getInputFunction()(game)))[2][1])
    end

    function calculateFruitDistance(game::Game)
        return sqrt((game.snake.head.x - game.fruit.x)^2 + (game.snake.head.y - game.fruit.y)^2)
    end

    function snakeFitness(cand::Chain)
        maxMovement = AI_MAX_MOVEMENT
        movCount = 0
        fitness = 0.0

        game = Game()

        while !isLost(game) && movCount < AI_MAX_MOVEMENT &&
                length(game.snake.body) < SNAKE_MAX_SIZE_TO_RANDOM
            nextMovement!(cand, game)
            prevSize = length(game.snake.body)

            distanceA = calculateFruitDistance(game)
    
            nextFrame!(game)

            distanceB = calculateFruitDistance(game)
            movCount += 1
            
            if distanceB < distanceA
                fitness += 1 / (distanceB + movCount)
                distanceA = distanceB
            else
                fitness -= 1 / (distanceB + movCount)
            end

            movCount = (1 - (length(game.snake.body) - prevSize)) * movCount
        end
        fitness += 5 * (length(game.snake.body) - 1) + 1

        return fitness
    end

    function createFloatNEATSet(maxPop::Int,
            deltaThreshold::Float64, c1::Float64, c2::Float64, c3::Float64,
            survivalRate::Float64, reproductionRate::Float64, biasMutationRate::Float64,
            weightMutationRate::Float64, toggleMutationRate::Float64,
            addNodeMutationRate::Float64, addConnectionMutationRate::Float64)
        setInputFunction!(FloatNEAT)
        net = Chain(NEATDense(20,4,sigmoid))
        set = NEAT.TrainingSet(net, net.layers[1], snakeFitness,
                        c1=c1, c2=c2, c3=c3, maxPopulation=maxPop,
                        survivalRate=survivalRate, deltaThreshold=deltaThreshold,
                        reproductionRate=reproductionRate, biasMutationRate=biasMutationRate,
                        weightMutationRate=weightMutationRate, 
                        toggleConnectionMutationRate=toggleMutationRate,
                        addNodeMutationRate=addNodeMutationRate, 
                        addConnectionMutationRate=addConnectionMutationRate)
        return set
    end

    function createGeneticSet(chain::Chain, popSize::Int,
        elitism::Int, crossoverDivisor::Int, mutationRate::Float64)
        set = Genetic.TrainingSet(chain, chain.layers, snakeFitness,
                    popSize=popSize, elitism=elitism, 
                    crossoverDivisor=crossoverDivisor, mutationRate=mutationRate)
        return set
    end

    @inline function createFloatGeneticSet(popSize::Int,
        elitism::Int, crossoverDivisor::Int, mutationRate::Float64)
        setInputFunction!(FloatMLP)
        return createGeneticSet(Chain(Dense(20,20, sigmoid), Dense(20, 4, sigmoid)), 
                    popSize, elitism, crossoverDivisor, mutationRate)
    end

    @inline function createBitGeneticSet(popSize::Int,
        elitism::Int, crossoverDivisor::Int, mutationRate::Float64)
        setInputFunction!(BitMLP)
        return createGeneticSet(Chain(BitDense(20,20), BitDense(20, 4, true)), 
                    popSize, elitism, crossoverDivisor, mutationRate)
    end

    @inline function Train!(set::Union{NEAT.TrainingSet,Genetic.TrainingSet}, genCount::Int)
        if set isa NEAT.TrainingSet
            NEAT.Train!(set, genCount)
        else
            Genetic.Train!(set, genCount)
        end
    end

end