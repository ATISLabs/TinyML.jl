module Snake
	include("./Snake/Model.jl")
	include("./Snake/Controller.jl")
	include("./Snake/View.jl")

	export nextFrame!, setDirection!, Game, randomizeFoodPosition!, Point2D
end