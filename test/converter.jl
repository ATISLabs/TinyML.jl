@testset "NEATConverter" begin
    println("Training NEAT")
    net = Chain(NEAT.NEATDense(2,1))
    func(net::Chain) = 1 / abs(net([3,2])[1] - 0.5)
    set = NEAT.TrainingSet(net, net.layers[1], func)
    NEAT.train!(set, maxFitness=500.0)

    println("Training BitNet")
    neatLayer = net.layers[1]
    newNet = Chain(BitDense(2, 1, true, identity))
    TinyML.NEATConverter.convert!(newNet, newNet.layers, neatLayer, 
        samples=1, maxFitness=100)
    TinyML.NEATConverter.random(in) = [3,2]
    @test round(newNet([3,2])[1], digits=1) == 0.5
end