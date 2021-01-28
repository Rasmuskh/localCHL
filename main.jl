
include("functions.jl")
include("LoadData.jl")

γ = 1/8
η = 0.01# 0.01
nNeurons = [784, 300, 300, 10]
Net = init_network(nNeurons, γ, η)
# Net = load("networks/Network_epoch_100.jld2")["Net"]

# Load dataset
trainSamples = 60000
testSamples = 10000
dataset = "MNIST"
xTrain, yTrain, xTest, yTest = loadData(dataset, trainSamples, testSamples)

nEpochs = 100
batchsize = 32
testBatchsize = min(10000, testSamples)
numIter = 8
outpath = "networks"
random_feedback = false

for epoch=1:nEpochs
    println("\nEpoch: $epoch")
    @time trainThreads(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, 1, numIter, ReLU, random_feedback, outpath)
end

