@testset "NEATConverter" begin
    @testset "DirectConverter" begin
        println("Training NEAT")
        net = Chain(NEAT.NEATDense(2,1))
        func(net::Chain) = 1 / abs(net([3,2])[1] - 0.5)
        set = NEAT.TrainingSet(net, net.layers[1], func, feedForward=true)
        NEAT.train!(set, maxFitness=500.0)

        new_net = Chain(NEATConverter.DirectConverter.to_denses(set)...)
        @test round(new_net([3,2])[1], digits=1) == 0.5
    end
end