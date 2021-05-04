
using Base: @kwdef
using Statistics
using LinearAlgebra
using Random; Random.seed!(342)
using LoopVectorization
# For saving and loading networks
using JLD2
using FileIO
# For formatting numeric print output
using Printf
# For loading datasets
using MLDatasets
# Flux dependencies
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Zygote
# function and structs from this project
include("LPOM.jl")
include("utilities.jl")
function main()

    # Initialize network
    nNeurons = [784, 512, 512, 10]
    activation = [relu, relu, identity] # Only supports relu and HS (Hard sigmoid)
    highLim = [Inf, Inf, Inf] # Upper clamping limit: ReLU: Inf, HS: 1
    init_mode =  "glorot_uniform" # Options are: "glorot_uniform" and "glorot_normal" 
    Net = init_network(nNeurons, highLim, init_mode)

    # Load a saved model
    # Net = load("networks/Net.jld2")["Net"] #Load old network

    #= Various arguments are stored in an Args struct.
    Default values can be overwritten by passing in keywords.=#
    args = Args(nEpochs=5, nNeurons=nNeurons, nLayers=length(nNeurons)-1,
                highLim=highLim, activation=activation, nInnerIterations=5)

    # Optimizer options can be found at: https://fluxml.ai/Flux.jl/stable/training/optimisers/
    optimizer = Descent(0.2)
    # optimizer = ADAM(0.0003)
    train(Net, args, optimizer)

return Net
end

Net = main();
println("Training finished")
