@testset "BitFlux" begin
    dense = BitDense(2,1)
    dense.W[1,1] = false
    dense.W[1,2] = true
    dense.b[1] = false
    net = Chain(dense)
    @test net(BitArray([false, true]))[1] == false
end