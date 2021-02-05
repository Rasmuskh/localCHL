
include("functions.jl")
include("LoadData.jl")

γ = 1/8
# η = 0.1
η = 0.001# 0.001

nNeurons = [784, 300, 300, 10]
Net = init_network(nNeurons, γ, η, energy, ReLU, get_z_fast!, get_∇w!, get_fb)
# Net = load("networks/Network_epoch_100.jld2")["Net"]

# Load dataset
trainSamples = 60000
testSamples = 10000
dataset = "MNIST"
xTrain, yTrain, xTest, yTest = loadData(dataset, trainSamples, testSamples)

nEpochs = 100
batchsize = 32
testBatchsize = min(10000, testSamples)
random_feedback = false
direct_random_feedback = false

numIter = 1
outpath = "./networks/MNIST_16-10_numIter_$numIter.jld2"
for epoch=1:nEpochs
    println("\nNET$numIter: Epoch: $epoch")
    @time trainThreads(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, 1, numIter, ReLU, random_feedback, direct_random_feedback, outpath)
end
