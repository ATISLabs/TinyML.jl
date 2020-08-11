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

const MAP_SIZE = 15
const SNAKE_POS = 8
const SNAKE_DIRECTION = UP


mutable struct Point2D
	x::Int
	y::Int
end

mutable struct Body
	head::Point2D
	body::Array{Point2D, 1}
	direction::Int

	function Body()
		s = new()

		s.body = Array{Point2D, 1}(undef, 0)
		s.head = Point2D(SNAKE_POS, SNAKE_POS)
		push!(s.body, Point2D(SNAKE_POS, SNAKE_POS))
		s.direction = SNAKE_DIRECTION

		return s
	end
end

mutable struct Game
	fruit::Point2D
	snake::Body
	lost::Bool

	pos::Int

	function Game()
		game = new()

		game.pos = 2
		game.snake = Body()
		game.fruit = Point2D(FOODPOS[1][1], FOODPOS[1][2])
		setLost!(game, false)

		return game
	end
end

@inline isLost(game::Game) = game.lost
@inline setLost!(game::Game, value::Bool) = game.lost = value

function setDirection!(game::Game, direction::Int)
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