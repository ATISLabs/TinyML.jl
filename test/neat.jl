@testset "NEAT" begin
    net = Chain(NEAT.NEATDense(2,1))
    func(net::Chain, none) = 1 / abs(net([3,2])[1] - 0.5)
    set = NEAT.TrainingSet(net, net.layers[1], func)
    NEAT.train!(set, maxFitness=500.0)
    @test round(net([3,2])[1], digits=1) == 0.5
end