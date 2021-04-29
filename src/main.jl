using BenchmarkTools

using Statistics
using LinearAlgebra

using JLD2; # Save and load network dictionaries
using FileIO; # Save and load network dictionaries
using Flux
include("LPOM.jl")
include("utilityFunctions.jl")
# Choose datatype
dType = Float32

# import stuff
#include("utilityFunctions.jl")
include("LoadData.jl")

#include("LPOM_BLAS_Batches.jl")
include("flux_mlp.jl")

function main()
    # nT = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
    # println(nT)
    LinearAlgebra.BLAS.set_num_threads(1)
    # nT = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
    # println(nT)

    # Define model architecture
    nNeurons = [784, 128, 128, 128, 10]
    HS(z) = Clamp(z, 0, 1.0)
    activation = [relu, relu, relu, identity]
    highLim = [Inf, Inf, Inf, Inf] # Upper clamping limit: ReLU: Inf, HardSigmoid: 1
    #activation = [HS, HS, HS]
    #highLim = [1.0, 1.0, 1.0]
    init_mode =  "glorot_uniform" # Options are: "glorot_uniform" and "glorot_normal" 
    Net0 = init_network(nNeurons, highLim, init_mode)
    # Load a saved model
    # Net = load("networks/MNIST.jld2")["Net"] #Load old network

    # Load dataset
    dataset = "MNIST"
    trainSamples = 60000
    testSamples = 10000
    xTrain, yTrain, xTest, yTest = loadData(dataset, trainSamples, testSamples)

    # Hyper parameters
    nOuterIterations = 2
    nInnerIterations = 5# for CIFAR 10-15 iterations seems to be needed
    nEpochs = 3
    batchsize = 64
    η = 0.5
    #
    testBatchsize = min(10000, testSamples)

    # Where to save the trained model
    outpath = "../networks/$(dataset).jld2"

    # Training loopn
    NetLPOM = deepcopy(Net0)
    for epoch=1:nEpochs
        println("\nEpoch: $epoch")
        @time train_LPOM_threads_V2(NetLPOM, xTrain, yTrain, xTest, yTest,
                               batchsize, testBatchsize, 1, η,
                               nOuterIterations, nInnerIterations, activation, outpath)
    end
    #BP trained MLP
    # use_CUDA = false
    # NetFlux = LPOM_to_Flux(deepcopy(Net0), activation)
    # NetFlux = train_flux(NetFlux, η, batchsize, nEpochs, use_CUDA)



return NetLPOM
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


