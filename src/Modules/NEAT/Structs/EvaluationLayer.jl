struct EvaluationLayer
    net::Ref{Candidate}
    σ::Function
end

EvaluationLayer(σ::Function) =
    EvaluationLayer(
        Ref{Candidate}(),
        σ
    )

#Getters and Setters
σ(layer::EvaluationLayer) = layer.σ

network(layer::EvaluationLayer) = layer.net[]
network!(layer::EvaluationLayer, candidate::Candidate) = 
    layer.net[] = candidate