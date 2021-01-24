
include("functions.jl")
include("LoadData.jl")

γ = 1/8#0.125
η = 0.4 # 0.0001
nNeurons = [784, 64, 64, 10]
Net = init_network(nNeurons, γ, η)
# Net = load("networks/Network_epoch_100.jld2")["Net"]

# Load dataset
trainSamples = 6000
testSamples = 1000
dataset = "MNIST"
xTrain, yTrain, xTest, yTest = loadData(dataset, trainSamples, testSamples)
nEpochs = 3
batchsize = 32
testBatchsize = min(10000, testSamples)
numIter = 15
outpath = "networks"
random_feedback = false

for epoch=1:nEpochs
    println("\nEpoch: $epoch")
    @time trainThreads(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, 1, numIter, ReLU, random_feedback, outpath)
end

# @time acc_train = predict(Net, xTrain, yTrain, batchsize, numIter, ReLU)
# println(acc_train)

# @time acc_test = predict(Net, xTest, yTest, batchsize, numIter, ReLU)
# println(acc_test)

