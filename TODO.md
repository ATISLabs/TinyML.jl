# TODO List
- [x] Game
- [ ] Tests
- [x] Neural Network
- [ ] Tests
- [x] Genetic Algorithm
- [ ] Tests
- [x] Binary Neural Network
- [ ] Tests
- [x] Binary Genetic
- [ ] Tests
- [x] GUI
- [ ] Tests
- [x] Matrix mult
- [ ] Tests
- [x] XOR Improvement
- [ ] Tests
- [x] Memory and CPU usage check and fix
- [ ] Tests
- [x] Multi-core
- [ ] Tests
- [x] Benchmarks
- [ ] Docs
- [ ] Readme
- [x] CNN
    - [x] CNN Base
    - [x] Input/Output switch
    - [x] Avoid padding
    - [x] Optimize stride to avoid i loop
    - [x] CNN Layers operations
        - [x] Conv
        - [x] MaxPool
        - [x] Flatten
        - [x] Activation
        - [x] MLP
    - [x] CNN Stride optimization
    - [x] CNN Allocation optimization
    - [x] CNN Test
    - [ ] Fix docs
    - [ ] Find an appropriate architecture for snake game


VERSION 2.0
- [x] Flux
    - [x] Flux MLP
    - [x] Flux bin MLP
    - [x] Compare speed
    - [x] Transpose AI functions to Flux compatible
    - [x] Flux genetic
        - [x] Fix Bit MLP
        - [ ] Fix Network names and gen count
    - [x] Flux CNN
- [x] Bin CNN
    - [x] Bin CNN Base
    - [ ] Bin CNN Layer
        - [ ] Conv
            - [ ] Conv Reshape for all kernel sizes and biggers inputs
        - [ ] Maxpool
    - [ ] Bin CNN Test


NEAT
- [x] Float NEAT
    - [x] Basic Structures
        - [x] NEAT forward fix order
            - [x] Recursive getOuput
                - [x] Loop fix
        - [x] NEAT Flux layer
        - [x] Network structure fix
    - [x] Selection
        - [x] Speciation
    - [x] Crossover
    - [x] Mutation
        - [x] Add node
        - [x] Add connection
        - [x] Mutate weights
- [ ] Genetic Refactoring
- [ ] Bit NEAT


FROM-SCRATCH
- [ ] Replace "BinarizedSnake" references to "BinarySnake"


BUG FIXES
- [ ] GUI
    - [ ] Resume training
    - [ ] Generation count
