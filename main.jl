
include("functions.jl")
include("LoadData.jl")

γ = 0.125
η = 0.4
nNeurons = [784, 64, 64, 10]
Net = init_network(nNeurons, γ, η)

# Load dataset
trainSamples = 60000
testSamples = 10000
dataset = "MNIST"
xTrain, yTrain, xTest, yTest = loadData(dataset, trainSamples, testSamples)
nEpochs = 10
batchsize = 50
numIter = 30
outpath = "networks"

for epoch=1:nEpochs
    println("\nEpoch: $epoch")
    @time trainThreads(Net, xTrain, yTrain, xTest, yTest, batchsize, 1, numIter, ReLU, outpath)
end

# @time acc_train = predict(Net, xTrain, yTrain, batchsize, numIter, ReLU)
# println(acc_train)

# @time acc_test = predict(Net, xTest, yTest, batchsize, numIter, ReLU)
# println(acc_test)

