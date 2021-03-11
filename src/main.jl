using BenchmarkTools
using LinearAlgebra

# nT = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
# println(nT)
LinearAlgebra.BLAS.set_num_threads(1)
# nT = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
# println(nT)

include("functions.jl")
include("LoadData.jl")

# Define model architecture
nNeurons = [784, 512, 512, 10]
Net = init_network(nNeurons)
activation = [ReLU, ReLU, ReLU]
#activation = [HardSigmoid, HardSigmoid, HardSigmoid, HardSigmoid]

# Load a saved model
# Net = load("networks/MNIST.jld2")["Net"] #Load old network

# Load dataset
dataset = "MNIST"
trainSamples = 60000
testSamples = 10000
xTrain, yTrain, xTest, yTest = loadData(dataset, trainSamples, testSamples)

# Hyper parameters
nOuterIterations = 2 # 2 seems to be insufficient to perform on par with BP
nInnerIterations = 5
nEpochs = 200
batchsize = 32
η = 0.4
testBatchsize = min(10000, testSamples)

# Where to save the trained model
outpath = "./networks/$(dataset).jld2"

# Training loop
for epoch=1:nEpochs
   println("\nEpoch: $epoch")
   @time trainThreads(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, 1, η, nOuterIterations, nInnerIterations, activation, outpath)
end





# x=xTrain[1];
# z = [randn(Float64, N) for N in Net.nNeurons[2:end]];
# w1x_plus_b1 = Net.w[1]*x+Net.b[1]
# forward!(z, Net, activation, w1x_plus_b1);
#z

# z = [randn(Float64, N) for N in nNeurons[2:end]]
# x=xTrain[1]
# w1x_plus_b1 = Net.w[1]*x+Net.b[1]
# # J = get_loss(Net, w1x_plus_b1, z, activation)
# denoms = [[sum(abs2,w[:,i]) for i=1:Net.nNeurons[layer+1]] for (layer, w) in enumerate(Net.w[2:end])]
# z = forward(z, Net, ReLU, w1x_plus_b1);
# z[end] = [yTrain[1]+1==k for k=1:10]
# println(z[1][1:5]')
# println(z[2][1:5]')
# #println(z[3][1:5]')
# println("========================")
# z = run_LPOM_inference(x, z, denoms, Net, 1, activation)
# println(z[1][1:5]')
# println(z[2][1:5]')
#println(z[3][1:5]')





