module Snake
	include("./Snake/Model.jl")
	include("./Snake/Controller.jl")
	include("./Snake/View.jl")

	export nextFrame!, setDirection!, randomizeFoodPosition!, getDrawingMatrix,
			isLost, setLost!
	export Game, Point2D
	export LEFT, UP, DOWN, RIGHT, MAP_SIZE
end