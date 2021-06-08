@testset "Ensemble" begin    

    @testset "Base" begin
        basenet = Chain(Dense(2,2), Dense(2,1))
        ensemble = TinyML.Ensembles.Ensemble(basenet, 3)
        net = Chain(TinyML.Ensembles.Boosting.BoostingEnsemble(ensemble))

        func(net::Chain) = 1 / abs(net([3,2])[1] - 0.5)
        set = Genetic.TrainingSet(net, (net.layers[1],))
        Genetic.train!(set, func, maxfitness=1000.0)
        @test round(net([3,2])[1], digits=1) == 0.5
    end

    @testset "Boosting" begin
        using TinyML
        using Flux

        dataset = [
            [3.0, 2.0] => 0.5,
            [2.0, 2.0] => 0.2,
            [3.0, 3.0] => 1.0
        ]

        basenet = Chain(Dense(2,20), Dense(20,1))
        ensemble = Ensemble(basenet, 2)
        net = Chain(Boosting.BoostingEnsemble(ensemble))

        set = Genetic.TrainingSet(net, (net.layers[1],))

        score(out::AbstractVector, y::Float64) = 1 - abs(out[1]-y)
        Genetic.train!(set, score, maxfitness=6.0, 
            loop=Boosting.generateloop(net.layers[1], dataset, 
                (out, y) -> abs(out[1] - y) < 0.05, 
                samplefraction=2))

        @test round(net([3,2])[1], digits=1) == 0.5
        @test round(net([2,2])[1], digits=1) == 0.2
        @test round(net([3,3])[1], digits=1) == 1
    end
end