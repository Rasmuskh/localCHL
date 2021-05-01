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
include("LPOM.jl")
include("utilityFunctions.jl")

@kwdef struct Args
    η::Float64 = 0.5
    batchsize::Int = 64
    test_batchsize::Int = 10000
    nOuterIterations::Int = 2
    nInnerIterations::Int = 5 # for CIFAR 10-15 iterations seems to be needed
    nEpochs::Int = 3
    activation::Vector{Function} = [relu, relu, relu, identity]
    highLim::Vector{Float64} = [Inf, Inf, Inf, Inf] # Upper clamping limit: ReLU: Inf, HardSigmoid: 1
    outpath::String = "../networks/Net.jld2"
    numThreads::Int = Threads.nthreads()
    nNeurons::Vector{Int32} = [784, 128, 128, 128, 10]
    nLayers::Int32 =  length(nNeurons)-1#4 # should match length(nNeurons)-1
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

    args = Args() # collect options in a struct for convenience
    N = Chain(Dense(args.nNeurons[1], args.nNeurons[2], relu),
          Dense( args.nNeurons[2],  args.nNeurons[3], relu),
          Dense( args.nNeurons[3],  args.nNeurons[4], relu),
          Dense( args.nNeurons[4],  args.nNeurons[5], identity))


    @time train_LPOM_threads(Net, args)

return Net
end

Net = main();
println("Training finished")


