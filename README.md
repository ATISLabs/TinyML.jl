# BinarySnake.jl
The classic Snake game in Julia, controlled by neural networks and trained by genetic algorithm

### Usage
```julia_repl
julia> using BinarySnake
julia> startBinarySnake()
```
It will open a GUI that allows controlling the parameters of genetic algorithm, train, save and load snake game training set. Also, it 
shows the evolution of the AI after each generation.


### What is the use of this branch?
This branch was created to help people on understanding basic evolutionary AI and binary neural networks by using genetic 
algorithm. Also, all functions are documented to help the learning of these algorithms in Julia!

### Why binary neural network converges faster than Float64?
First, keep in mind that binary neural network (both inputs and weights in this case) shall be represented by only two values, true 
or false, and because of that, it loses information in reason of not being able to represent any values between -inf and +inf. In some 
cases, losing information isn't something good, but in this case that representation is able to reduce the search space on finding 
correct moves for the game, what helps genetic algorithm convergence. Also, using bit-wise operations, it is able to have 64 bits into 
a single Float64 variable, which allows it to execute a single operation for 64 bit variables, incresing code speed and memory 
consumption.

Have fun!
