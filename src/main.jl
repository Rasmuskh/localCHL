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
include("utilityFunctions.jl")

@kwdef struct Args
    # Hyper parameters
    batchsize::Int = 64
    test_batchsize::Int = 10000
    nOuterIterations::Int = 2
    nInnerIterations::Int = 5 # use ~10-15 for CIFAR10
    nEpochs::Int = 10
    # Network specific parameters
    nNeurons::Vector{Int32} = [784, 128, 128, 128, 10]
    activation::Vector{Function} = [relu, relu, relu, identity]
    highLim::Vector{Float64} = [Inf, Inf, Inf, Inf] # Upper clamping limit: ReLU: Inf, HardSigmoid: 1
    nLayers::Int32 =  length(nNeurons)-1 # should match length(nNeurons)-1
    # misc parameters
    outpath::String = "../networks/Net.jld2"
    numThreads::Int = Threads.nthreads()

end

function main()
    # Define model architecture
    nNeurons = [784, 128, 128, 128, 10]
    HS(z) = Clamp(z, 0, 1.0)
    activation = [relu, relu, relu, identity]
    highLim = [Inf, Inf, Inf, Inf] # Upper clamping limit: ReLU: Inf, HardSigmoid: 1

    # Initialize network
    init_mode =  "glorot_uniform" # Options are: "glorot_uniform" and "glorot_normal" 
    Net = init_network(nNeurons, highLim, init_mode)
    # Load a saved model
    # Net = load("networks/Net.jld2")["Net"] #Load old network

    #= Various arguments are stored in an Args struct. Default values can be overwritten by passing in
    keywords. Optimizer options can be found at: https://fluxml.ai/Flux.jl/stable/training/optimisers/=#
    args = Args(nEpochs=3)
    optimizer = Descent(0.2)
    #optimizer = ADAM(0.0003)
    train(Net, args, optimizer)

return Net
end

Net = main();
println("Training finished")


