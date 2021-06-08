function loop!(set::TrainingSet, fitness::Function)
    evaluate!(set, fitness)
    selection_best!(set)
    crossover_clone!(set)
    mutation_rand!(set)
end

function train!(set::TrainingSet, fitness::Function; 
        genlimit::Int=typemax(Int), maxfitness::Float64=Inf64,
        loop::Function=loop!,
        print::Bool = true)

    gen = 0
    while gen < genlimit && bestfitness(set) < maxfitness
        loop(set, fitness)
        if(print) gen += 1; print_gen(gen, bestfitness(set)) end
    end
    update!(set)
    set
end

#= Common/Other =#
@inline bestfitness(set::TrainingSet) = fitness(first(set))
@inline print_gen(gen::Int, fit::Float32) =
    @info "Gen: $(gen) -- Fitness: $(fit)"

@inline randcopylayer(dense::Dense) = 
    Dense(size(weight(dense), 2), size(weight(dense), 1), activation(dense))
@inline randcopylayer(dense::BitDense{T}) where {T} = 
    BitDense{T}(size(weight(dense), 2), size(weight(dense), 1), activation(dense))

function update!(old::Union{Dense, BitDense}, new::Union{Dense, BitDense})
    weight(old) .= weight(new)
    bias(old) .= bias(new)
end

function update!(set::TrainingSet)
    sort!(set)
    for i in indexes(set)
        update!(layers(chain(set))[i], layers(network(first(set)))[i])
    end
end

@inline rand_weight() = Float32(rand(Uniform(-1,1)))
@inline rand_bias() = Float32(rand(Uniform(-1,1)))