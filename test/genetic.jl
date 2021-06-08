@testset "Genetic" begin    
    net = Chain(Dense(2,2), Dense(2,1))
    func(net::Chain) = 1 / abs(net([3,2])[1] - 0.5)
    set = Genetic.TrainingSet(net, (net.layers[2],))
    Genetic.train!(set, func, maxfitness=100.0)
    @test round(net([3,2])[1], digits=1) == 0.5

    net = Chain(BitDense(2,20), BitDense(20,1, true))
    func2(net::Chain) = 1 / abs(net([3,2])[1] - 0.5)
    set = Genetic.TrainingSet(net, (net.layers[2],))
    Genetic.train!(set, func2, maxfitness=100.0)
    @test round(net([3,2])[1], digits=1) == 0.5
end