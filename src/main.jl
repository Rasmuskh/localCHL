using BenchmarkTools

using Statistics
using LinearAlgebra
using JLD2; # Save and load network dictionaries
using FileIO; # Save and load network
using Printf # For formatting numeric output
using Random; Random.seed!(32); rng = MersenneTwister(13)
using LinearAlgebra
using MLDatasets
using Base: @kwdef
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Zygote
include("LPOM.jl")
include("utilities.jl")

function main()

    # Initialize network
    nNeurons = [784, 128, 128, 128, 10]
    activation = [relu, relu, relu, identity]
    highLim = [Inf, Inf, Inf, Inf] # Upper clamping limit: ReLU: Inf, HardSigmoid: 1
    init_mode =  "glorot_uniform" # Options are: "glorot_uniform" and "glorot_normal" 
    Net = init_network(nNeurons, highLim, init_mode)

    # Load a saved model
    # Net = load("networks/Net.jld2")["Net"] #Load old network

    #= Various arguments are stored in an Args struct. Default values can be overwritten
    by passing in keywords.=#
    args = Args(nEpochs=3, nNeurons=nNeurons, nLayers=length(nNeurons)-1,
                highLim=highLim, activation=activation)

    # Optimizer options can be found at: https://fluxml.ai/Flux.jl/stable/training/optimisers/=#
    optimizer = Descent(0.2)
    #optimizer = ADAM(0.0003)
    train(Net, args, optimizer)

return Net
end

Net = main();
println("Training finished")


