struct StackedModel
    models::Array{Chain, 1}
    meta::Metalearner
end

function StackedModel(learner, models...)
    StackedModel(models, learner(models))
end

(stack::StackedModel)(input) = meta(stack)(vcat(models(stack).(input)))

meta(stack::StackedModel) = stack.meta
models(stack::StackedModel) = stack.models