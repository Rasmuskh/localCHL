using BenchmarkTools
include("functions.jl")
include("LoadData.jl")

nNeurons = [784, 128, 128, 128, 10]
Net = init_network(nNeurons)
# Net = load("networks/Network_epoch_100.jld2")["Net"] #Load old network

# Load dataset
trainSamples = 60000
testSamples = 10000
dataset = "MNIST"
xTrain, yTrain, xTest, yTest = loadData(dataset, trainSamples, testSamples)


activation = ReLU
numIter = 2

nEpochs = 100
batchsize = 64
η = 0.01
testBatchsize = min(10000, testSamples)
outpath = "./networks/$(dataset).jld2"
for epoch=1:nEpochs
    println("\nEpoch: $epoch")
    @time trainThreads(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, 1, η, numIter, activation, outpath)
end


# x=xTrain[1]; 
# z = [zeros(Float64, N) for N in Net.nNeurons[2:end]];
# forward!(z, Net, ReLU, Net.w[1]*x+Net.b[1]);
# z

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




# println(z[1][1:12]')
# run_BCD_LPOM!(x, z[1], z[2],
#               Net.w[1], Net.w[2], Net.b[1], Net.b[2],
#               similar(z[1]), similar(z[1]), denoms[1], similar(z[2]), similar(z[2]))
# println(z[1][1:5]')
# println(z[2][1:5]')
# println(z[3][1:5]')
# @btime run_LPOM_inference!(z, denoms, Net, numIter)
# println(z[1][1:5]')
# println(z[2][1:5]')
# println(z[3][1:5]')


# println(J)
# get_z!(w1x_plus_b1, z, Net, activation, numIter, clamped)

# ∇w = [zeros(Float64, (nNeurons[i+1], nNeurons[i])) for i = 1:Net.nLayers-1]
# ∇b = [zeros(Float64, (nNeurons[i+1])) for i = 1:Net.nLayers-1]
# get_∇w!(∇w, ∇b, x, w1x_plus_b1, z, activation, Net)
# Net.w -= η*∇w
# Net.b -= η*∇b
# w1x_plus_b1 = Net.w[1]*x+Net.b[1]
# get_z!(w1x_plus_b1, z, Net, activation, numIter, clamped)
# J = get_loss(Net, w1x_plus_b1, activation)
# println(J)
# for t=1:20
#     get_z!(w1x_plus_b1, z, Net, activation, numIter, clamped)
#     println(get_loss(Net, w1x_plus_b1, activation))
# end






