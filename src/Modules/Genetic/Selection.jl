function selection_best!(tset::TrainingSet)
    sort!(tset)
end

function swap_indexes!(arr::Array{Candidate, 1}, index1::Integer, index2::Integer)
    aux = arr[index1]
    arr[index1] = arr[index2]
    arr[index2] = aux
end
