using BenchmarkTools

using Statistics
using LinearAlgebra

using JLD2; # Save and load network dictionaries
using FileIO; # Save and load network dictionaries
using MLDatasets
using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
include("LPOM.jl")
include("utilityFunctions.jl")

function main()
    # Define model architecture
    nNeurons = [784, 50, 50, 50, 10]
    HS(z) = Clamp(z, 0, 1.0)
    activation = [relu, relu, relu, identity]
    highLim = [Inf, Inf, Inf, Inf] # Upper clamping limit: ReLU: Inf, HardSigmoid: 1

    # Initialize network
    init_mode =  "glorot_uniform" # Options are: "glorot_uniform" and "glorot_normal" 
    Net = init_network(nNeurons, highLim, init_mode)
    # Load a saved model
    # Net = load("networks/MNIST.jld2")["Net"] #Load old network

    # Hyper parameters
    nOuterIterations = 2
    nInnerIterations = 5# for CIFAR 10-15 iterations seems to be needed
    nEpochs = 3
    batchsize = 64
    test_batchsize = 10000
    η = 0.5

    # Train the network
    outpath = "../networks/Network.jld2"
    numThreads = Threads.nthreads()
    @time train_LPOM_threads(Net, batchsize, test_batchsize, nEpochs, η,
                             nOuterIterations, nInnerIterations,
                             activation, outpath, numThreads)
return Net
end

Net = main();
println("Training finished")





# Training loop
# for epoch=1:nEpochs
#     println("\nEpoch: $epoch")
#     @time trainBatches(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, 1, η, nOuterIterations, nInnerIterations, activation, outpath)
# end





# X = reduce(hcat, xTrain[1:batchsize])
# Z = [randn(Float64, (N,batchsize)) for N in Net.nNeurons[2:end]];
# w1x_plus_b1 = Net.w[1]*X.+Net.b[1]
# forward!(Z, Net, activation, w1x_plus_b1);
# k=2
# i=3
# C = Net.w[k+1]*Z[k]
# c = C[:,1]
# c2 = c - Net.w[k+1][:,i].*Z[k][i,1]

# C = Net.w[k+1]*Z[k]
# for col=1:batchsize
#     C[:,col] -= Net.w[k+1][:,i].*Z[k][i,col]
# end


# k=2
# i=3
# a = Net.w[k]* Z[k-1] .+ Net.b[k]
# b =  Net.w[k+1]'*(Z[k+1] .- Net.b[k+1])
# wTwZ = Net.w[k+1]'*Net.w[k+1]*Z[k]
# U = Net.w[k+1]'*ReLU(-Net.w[k+1]*Z[k] .- Net.b[k+1])
# V = Net.w[k+1]'*ReLU(Net.w[k+1]*Z[k] .+ Net.b[k+1] .-1)

# Z_old = Z[k][i,:]
# w1 = view(Net.w[k+1], :, i)
# c=Net.w[k+1]*Z[k]
# for i=1:batchsize
#     for j = 1:length(Z[k][:,1])
#         c[j,i] -= Net.w[k+1][:,j]*Z[j,i]
#     end
# end
# c_sub = Net.w[k+1][:,i]*Z[k][i,:]
# @btime Z[k][i,:] = view(a, i, :) + view(b, i, :) + view(U, i, :) + view(V, i, :)
# @btime Z[k][i,:] .= a[i,:] .+ b[i,:] .+ U[i,:] .+ V[i,:]


# J = get_loss2(Net, w1x_plus_b1, Z, activation)


