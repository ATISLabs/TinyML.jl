@testset "BitFlux" begin
    dense = BitDense(2,1)
    dense.weight[1,1] = false
    dense.weight[1,2] = true
    dense.bias[1] = false
    net = Chain(dense)
    @test net(BitArray([false, true]))[1] == false
end