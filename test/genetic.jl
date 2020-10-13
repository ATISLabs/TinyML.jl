@testset "Genetic" begin    
    net = Chain(Dense(2,2), Dense(2,1))
    func(net::Chain) = 1 / abs(net([3,2])[1] - 0.5)
    set = Genetic.TrainingSet(net, (net.layers[2],), func)
    Genetic.train!(set, 100)
    @test round(net([3,2])[1], digits=1) == 0.5

    net = Chain(BitDense(2,20), BitDense(20,1, true))
    func(net::Chain) = 1 / abs(net([3,2])[1] - 0.5)
    set = Genetic.TrainingSet(net, (net.layers[2],), func)
    Genetic.train!(set, 100)
    @test round(net([3,2])[1], digits=1) == 0.5
end