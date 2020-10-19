# TinyML.jl
A package focused on using bit operations with deep learning techniques for improving performance and reducing memory consumption in order to be executed on low-end hardware.

## Features
* **BitFlux:** A module containing bit layers and functions to be used with Flux. Also, its syntax is similar to Flux syntax.
* **NEAT:** Implementation of the NEAT algorithm of reinforcement learning.
* **Genetic:** The basic genetic algorithm of reinforcement learning.

## What can I do with it?
Check this simple example bellow:
```julia
using Flux
using TinyML

layers = (BitDense(2,1),)
chain = Chain(layers...)
fitness(chain::Chain) = 1/(abs(chain([3,2]-0.5)))

set = Genetic.TrainingSet(chain, layers, fitness)
Genetic.train!(set, maxFitness=100.0)
```
With only a few lines you can create a model trained with reinforcement learning methods. It is simple and intuitive to get it!

## Why should I care for this package?
Even if you are not seeking for running your models in low-end hardware, this package may fit for your purposes if you want faster convergence or binary weights in your model.

By using binary weights, the search space is reduced to only two values, 0 and 1, instead of regular models who are comprehended between -Inf and +Inf.

## What benefits will I gain by using this package?
* **Speed:** By using BitDense and BitArray input, the performance can be improved up to 14x according to our tests. Check an example bellow:
```
binput = bitrand(640)
bdense = BitDense(640, 64)
--------------------------
Mean time: 841.554 ns

finput = rand(640)
fdense = Dense(640, 64)
--------------------------
Mean time: 9.662μs
```
* **Memory consumption:** By using BitDense you can achieve up to 32x less memory usage when comparing to regular Dense. Check an example bellow:
```
fdense = Dense(640, 64)
--------------------------
Size: 5228 bytes

bdense = BitDense(640, 64)
--------------------------
Size: 164192 bytes
```
* **Convergence:** When training a model using BitDense instead of Dense, we could improve the convergence time by reducing the search space. Check an example bellow:
```
Float Snake: 16.05 generations
Bit Snake: 5.24 generations
```