"""
    activationSigmoid(number::Number)

Sigmoid function, it is equal to (1/(1+exp(-number)))

# Example
```julia_repl
julia> activationSigmoid(0)
0.5
```
"""
@inline activationSigmoid(number::Number) = 1 / (1 + exp(-number))

"""
    activationNone(number::Number)

It doesn't do anything, it only returns the same input number

# Example
```julia_repl
julia> activationNone(5)
5
```
"""
@inline activationNone(number::Number) = number
