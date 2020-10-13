function randomizeFoodPosition!(game::Game)
	#if length(game.snake.body) > SNAKE_MAX_SIZE_TO_RANDOM
		game.fruit.x = rand(1:MAP_SIZE)
		game.fruit.y = rand(1:MAP_SIZE)
	#=else
		game.fruit.x = FOODPOS[game.pos][1]
		game.fruit.y = FOODPOS[game.pos][2]
		game.pos+=1
		if game.pos == length(FOODPOS)
			game.pos = 1
		end
	end=#
end

function getDrawingMatrix(game::Game)
		matrix = Array{Char, 2}(undef, MAP_SIZE, MAP_SIZE)
		fill!(matrix, DRAW_MATRIX_VOID)

		for piece in game.snake.body
			matrix[piece.y, piece.x] = DRAW_MATRIX_SNAKE
		end

		matrix[game.fruit.y, game.fruit.x] = DRAW_MATRIX_FRUIT

		return matrix
end

function nextFrame!(game::Game)
	if !isLost(game)
		#Lost the game when the snake reaches map limits
		if game.snake.direction == UP
			if game.snake.head.y != 1
				game.snake.head.y -= 1
			else
				setLost!(game, true)
			end
		elseif game.snake.direction == DOWN
			if game.snake.head.y != MAP_SIZE
				game.snake.head.y += 1
			else
				setLost!(game, true)
			end
		elseif game.snake.direction == LEFT
			if game.snake.head.x != 1
				game.snake.head.x -= 1
			else
				setLost!(game, true)
			end
		else
			if game.snake.head.x != MAP_SIZE
				game.snake.head.x += 1
			else
				setLost!(game, true)
			end
		end

		if game.snake.head.x == game.fruit.x &&
			game.snake.head.y == game.fruit.y
			randomizeFoodPosition!(game)
		else
			pop!(game.snake.body)
		end

		pushfirst!(game.snake.body, deepcopy(game.snake.head))

		for piece in game.snake.body[2:length(game.snake.body)]
			if game.snake.head.x == piece.x &&
				game.snake.head.y == piece.y
				setLost!(game, true)
			end
		end
	end
end