struct Ensemble
    classifiers::Vector{Chain}
    combiner::Function
end

Ensemble(basetopology::Chain,
        nofweakclassifiers::Int;
        combiner::Function= mean_combiner
       ) =
    Ensemble([deepcopy(basetopology) for i in 1:nofweakclassifiers]
        , combiner)

Ensemble(classifiers::Vector{Chain}, ensemble::Ensemble) = 
    Ensemble([classifier for classifier in classifiers], combiner(ensemble))

@inline (ensemble::Ensemble)(input::AbstractArray) =
    combiner(ensemble)([classifier(input) for classifier in classifiers(ensemble)])

mean_combiner(input::Vector{T}) where T <: AbstractArray =
    map(+, input...) ./ length(input)

#Getters
@inline classifiers(ensemble::Ensemble) = ensemble.classifiers
@inline combiner(ensemble::Ensemble) = ensemble.combiner    

@inline Base.length(ensemble::Ensemble) = length(classifiers(ensemble))

# Displays 
function Base.show(io::IO, ensemble::Ensemble)
    print(io, "$(length(ensemble))-Ensemble")
end
function Base.display(ensemble::Ensemble)
    print("$(length(ensemble))-classifier Ensemble\n",
        ["$(classifier)\n" for classifier in classifiers(ensemble)]...)
end