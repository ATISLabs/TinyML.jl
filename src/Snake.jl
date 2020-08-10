"""
	mutable struct Point2D

Stores x and y position of somethings

# Structure
mutable struct Point2D
	x::Int
	y::Int
end

# Initialization
Point2D(x::Int, y::Int)

# Example
```julia_repl
julia> Point2D(5, 3)
```
It will make a new Point2D with X coordinade equals to 5 and
Y coordinade equals to 3
"""
mutable struct Point2D
	x::Int
	y::Int
end

"""
	mutable struct Snake

Stores snake data

# Structure
mutable struct Snake
	head::Point2D
	body::Array{Point2D, 1}
	direction::Int
end

# Initialization
```julia
Snake(x::Int, y::Int, direction::Int)
```

**For initialization, you shall use one of the following constants for direction:**
- UP
- RIGHT
- LEFT
- DOWN
"""
mutable struct Snake
	head::Point2D
	body::Array{Point2D, 1}
	direction::Int

	function Snake(x::Integer, y::Integer, direction::Int)
		s = new()

		s.body = Array{Point2D, 1}(undef, 0)
		s.head = Point2D(x, y)
		push!(s.body, Point2D(x, y))
		s.direction = direction

		return s
	end
end

"""
	mutable struct SnakeGame

Stores the Game data

# Structure
```julia
mutable struct SnakeGame
	fruit::Point2D
	snake::Snake
	mapW::Int
	mapH::Int
	lost::Bool

	pos::Int
end
```
- fruit: Stores the current fruit position
- snake: Stores the snake data
- mapW: Stores the map width 
- mapH: Stores the map height
- lost: Stores the current state of game
- pos: Stores the current index of the FOODPOS array

ps. FOODSPOS is an Array containing positions for the fruit, 
until the game starts to randomize its position

# Initialization
```julia
SnakeGame(mapWidth::Int, mapHeight::Int, snakeXPosition::Int, snakeYPosition::Int, snakeDirection::Int)
```

For initialization, you shall use one of the following constants for 'snakeDirection':
- UP
- RIGHT
- LEFT
- DOWN
"""
mutable struct SnakeGame
	fruit::Point2D
	snake::Snake
	mapW::Int
	mapH::Int
	lost::Bool

	pos::Int

	function SnakeGame(mapW::Integer, mapH::Integer, snakeX::Integer, 
		snakeY::Integer, snakeDirection::Int)
		# Checks map bounds
		if(mapW < 2 || mapH < 2)
			error("Map too small")
		elseif(snakeX > mapW || snakeX < 1 ||
			snakeY > mapH || snakeY < 1)
			error("Snake position not in map boundaries")
		end

		game = new()

		#
		game.pos = 2

		game.mapW = mapW
		game.mapH = mapH
		game.snake = Snake(snakeX, snakeY, snakeDirection)
		game.fruit = Point2D(FOODPOS[1][1], FOODPOS[1][2])
		game.lost = false

		return game
	end
end

"""
	snakeSetDirection(game::SnakeGame, direction::Int)

Sets the snake direction in the game.
It also checks if the snake is trying to change its direction
to a direction where a piece of its body is located

# Example
```julia_repl
julia> snakeSetDirection(game, UP)
```
**For direction change, you shall use one of the following constants:**
- UP
- RIGHT
- LEFT
- DOWN
"""
function snakeSetDirection(game::SnakeGame, direction::Int)
	if length(game.snake.body) > 1
		yEqual = game.snake.body[2].y == game.snake.head.y
		xEqual = game.snake.body[2].x == game.snake.head.x

		if yEqual && game.snake.body[2].x < game.snake.head.x && direction == LEFT
			return
		elseif yEqual && game.snake.body[2].x > game.snake.head.x && direction == RIGHT
			return
		elseif xEqual && game.snake.body[2].y < game.snake.head.y && direction == UP
			return
		elseif xEqual && game.snake.body[2].y > game.snake.head.y && direction == DOWN
			return
		end
	end
		game.snake.direction = direction
end

"""
	snakeRandomizeFoodPosition(game::SnakeGame)

Change the food position to a random one

Example
```julia_repl
julia> snakeRandomizeFoodPosition(game)
```
"""
function snakeRandomizeFoodPosition(game::SnakeGame)
	if length(game.snake.body) > SNAKE_MAX_SIZE_TO_RANDOM
		game.fruit.x = rand(1:game.mapW)
		game.fruit.y = rand(1:game.mapH)
	else
		game.fruit.x = FOODPOS[game.pos][1]
		game.fruit.y = FOODPOS[game.pos][2]
		game.pos+=1
		if game.pos == length(FOODPOS)
			game.pos = 1
		end
	end
end

"""
	snakeGetDrawMatrix(game::SnakeGame)

Gets a Char matrix representing the game grid where
'X' represents a piece of snake body, 'O' represents 
a fruit and ' ' represents nothing.

# Example
```julia_repl
julia> snakeGetDrawMatrix(game)
' ' ' ' ' 'O' ' 
' ' ' 'X' ' ' ' 
' ' ' 'X' ' ' ' 
' 'X' 'X' ' ' ' 
```
"""
function snakeGetDrawMatrix(game::SnakeGame)
		matrix = Array{Char, 2}(undef, game.mapW, game.mapH)
		fill!(matrix, DRAW_MATRIX_VOID)

		for piece in game.snake.body
			matrix[piece.y, piece.x] = DRAW_MATRIX_SNAKE
		end

		matrix[game.fruit.y, game.fruit.x] = DRAW_MATRIX_FRUIT

		return matrix
end

"""
	snakeNextFrame(game::SnakeGame)

Processes the next state of game by doing a snake move,
snake eating fruit and snake dying.

- If the Snake reaches the map boundaries, it sets the 'lost' variable in game to true
- If the Snake reaches the fruit, it increases the size of the snake by adding a new piece of
its body at the position where the fruit were and randomizes the fruit position
- It does the snake movement by using 'pop' and 'pushfirst' operations, which means the snake
body is a queue

# Example
```julia_repl
julia> snakeNextFrame(game)
```
"""
function snakeNextFrame(game::SnakeGame)
	if !game.lost
		#Teleports the snake when reaches map limits
		#=if game.snake.direction == UP
			game.snake.head.y = game.snake.head.y != 1 ? (game.snake.head.y - 1) : game.mapH
		elseif game.snake.direction == DOWN
			game.snake.head.y = game.snake.head.y != game.mapH ? (game.snake.head.y + 1) : 1
		elseif game.snake.direction == LEFT
			game.snake.head.x = game.snake.head.x != 1 ? (game.snake.head.x - 1) : game.mapW
		else
			game.snake.head.x = game.snake.head.x != game.mapW ? (game.snake.head.x + 1) : 1
		end
		=#
		#Lost the game when the snake reaches map limits
		if game.snake.direction == UP
			if game.snake.head.y != 1
				game.snake.head.y -= 1
			else
				game.lost = true
			end
		elseif game.snake.direction == DOWN
			if game.snake.head.y != game.mapH
				game.snake.head.y += 1
			else
				game.lost = true
			end
		elseif game.snake.direction == LEFT
			if game.snake.head.x != 1
				game.snake.head.x -= 1
			else
				game.lost = true
			end
		else
			if game.snake.head.x != game.mapW
				game.snake.head.x += 1
			else
				game.lost = true
			end
		end

		if game.snake.head.x == game.fruit.x &&
			game.snake.head.y == game.fruit.y
			snakeRandomizeFoodPosition(game)
		else
			pop!(game.snake.body)
		end

		pushfirst!(game.snake.body, deepcopy(game.snake.head))

		for piece in game.snake.body[2:length(game.snake.body)]
			if game.snake.head.x == piece.x &&
				game.snake.head.y == piece.y
				game.lost = true
			end
		end
	end
end