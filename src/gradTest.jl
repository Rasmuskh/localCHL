using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Flux.Losses: mse
using MLDatasets
using LinearAlgebra
using Statistics
using Random; Random.seed!(3323); rng = MersenneTwister(12333)
include("plottingFunctions.jl")

function getdata(batchsize, trainsamples, testsamples)

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtrain = xtrain[:,:,1:trainsamples]; ytrain = ytrain[1:trainsamples]
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
    xtest = xtest[:,:,1:testsamples]; ytest = ytest[1:testsamples]

    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=batchsize)

    return train_loader, test_loader
end


function loss_and_accuracy(data_loader, Net)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        #x, y = device(x), device(y)
        #pred = Net(x)
        # ls += logitcrossentropy(model(x), y, agg=sum)
        ls += mse(Net(x), y, agg=sum)
        acc += sum(onecold(cpu(Net(x))) .== onecold(cpu(y)))
        num +=  size(x, 2)
    end
    return ls / num, acc / num
end

function train_BP(Net, batchsize, opt, nEpochs, trainsamples, testsamples)

    train_loader, test_loader = getdata(batchsize, trainsamples, testsamples)

    θ = params(Net)

    ## Training
    for epoch in 1:nEpochs
        t1 = time()
        for (x, y) in train_loader
            ∇θ = gradient(() -> mse(Net(x), y), θ) # compute gradient
            Flux.Optimise.update!(opt, θ, ∇θ) # update parameters
        end

        # Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, Net)
        test_loss, test_acc = loss_and_accuracy(test_loader, Net)
        #plot_filters(params(Net[1])[1], 8, 8, 800, 800, 28, 28, "/home/rasmus/Documents/localCHL/output/resurectionNet/FiltersBP/epoch$(epoch).png")
        t2 = time()
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
        println("  Runtime: $(t2-t1)")
    end
    return Net
end


"""Probably not needed anymore"""
function get_z1_hat(x, W0, b0, β, ∇z1_free, activation)
    z1_hat = activation.(W0*x .+b0 .-β*∇z1_free)
    return z1_hat
end

"""Probably not needed anymore"""
function get_∇θ1(x, z1_hat, z1_free, W0, b0, β, batchsize)
    ∇b1 = (z1_free .- z1_hat)/β
    ∇W1 = (∇b1*x')

    ∇b1 = sum(∇b1, dims=2)#/batchsize
    #∇W1 /= batchsize
    return (∇b1, ∇W1)
end

function f(X, z1_free, ∇z1_free, W1, b1, activation)
    #TODO: Reduce number of allocations
    batchsize = (size(X)[2])
    A = W1*X .+ b1
    a = [ai for ai in eachcol(A)]
    x_batch = [col for col in eachcol(X)]
    z1_free_batch = [col for col in eachcol(z1_free)]
    ∇z1_free_batch = [col for col in eachcol(∇z1_free)]
    z1_hat_batch = [similar(b1) for k=1:batchsize]#similar(z1_free_batch)
    ∇W1 = similar(W1)
    ∇b1 = similar(b1)
    ∇W1_batch = [similar(W1) for k=1:batchsize]
    ∇b1_batch = [similar(b1) for k=1:batchsize]

    β_best_Arr = [similar(b1) for k=1:batchsize]
    LinearAlgebra.BLAS.set_num_threads(1)
    Threads.@threads for i=1:batchsize # Loop over datapoints
        for n=1:length(b1) # Loop over neurons
            β = 10.0;  β_best_Arr[i][n] = β
            z1_hat_batch[i][n] = activation(a[i][n] - β*∇z1_free_batch[i][n])
            E1_hat = ((a[i][n] - z1_hat_batch[i][n])^2)
            E1_free = ((a[i][n] - z1_free_batch[i][n])^2)
            ΔE_max = (E1_hat - E1_free)/β
            E1_hat_max = E1_hat
            #Choose β to maximize ΔE = E_hat - E_free
            #TODO: Optimize this section
            for t=1:8
                β *= 2
                z_dummy = activation(a[i][n] - β*∇z1_free_batch[i][n])
                E1_hat = (a[i][n] - z1_hat_batch[i][n])^2
                ΔE = (E1_hat - E1_free)/β
                if ΔE>=ΔE_max
                    ΔE_max = ΔE
                    z1_hat_batch[i][n] = z_dummy
                    β_best_Arr[i][n] = β
                else
                    break
                end

            end
            ∇b1_batch[i][n] =  (z1_free_batch[i][n] .- z1_hat_batch[i][n])/β_best_Arr[i][n]
            ∇W1_batch[i][n,:] =  ∇b1_batch[i][n]*x_batch[i]'
        end
    end
    LinearAlgebra.BLAS.set_num_threads(8)
    β_av = sum(reduce(hcat,β_best_Arr))/length(b1)
    β_std = std(reduce(hcat,β_best_Arr))/length(b1)
    ∇W1 = mean(∇W1_batch)
    ∇b1 = mean(∇b1_batch)

    # For debugging
    # β = 4096.0; β_av = 0; β_std = 0
    # z1_hat = get_z1_hat(X, W1, b1, β, ∇z1_free, activation)
    # ∇b1, ∇W1 = get_∇θ1(X, z1_hat, z1_free, W1, b1, β, batchsize)

    return (∇b1, ∇W1, β_av, β_std)
end


function train_hybrid(Net, batchsize, opt, nEpochs,  trainsamples, testsamples)

    train_loader, test_loader = getdata(batchsize, trainsamples, testsamples)

    θ = params(Net)
    activation = relu
    β_av_hist = zeros(dType, nEpochs)
    β_std_hist = zeros(dType, nEpochs)
    # Training
    for epoch in 1:nEpochs
        t1 = time()
        β_av = 0.0
        β_std = 0.0
        for (x, y) in train_loader

            z1_free = Net[1](x)
            ∇θ = gradient(() -> mse(Net(x), y), θ) # compute gradient

            grad = gradient(() -> mse(Net[2:end](z1_free), y), params([z1_free]))
            ∇z1_free = grad[z1_free]
            ∇b1, ∇W1, β_av_batch, β_std_batch = f(x, z1_free, ∇z1_free, θ[1], θ[2], activation)
            β_av += β_av_batch
            β_std += β_std_batch

            ∇θ[θ[1]][:,:] = ∇W1
            ∇θ[θ[2]][:] = ∇b1
            Flux.Optimise.update!(opt, θ, ∇θ) # update parameters
        end
        β_av_hist[epoch] = β_av/trainsamples # get average batch mean(β)
        β_std_hist[epoch] = β_std/trainsamples # get average batch std(β)
        # Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, Net)
        test_loss, test_acc = loss_and_accuracy(test_loader, Net)
        #plot_filters(params(Net[1])[1], 8, 8, 800, 800, 28, 28, "/home/rasmus/Documents/localCHL/output/resurectionNet/FiltersResurection/epoch$(epoch).png")
        t2 = time()
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
        println("  β_av = $(β_av_hist[epoch]), β_std = $(β_std_hist[epoch])")
        println("  Runtime: $(t2-t1)")

    end


    return Net, β_av_hist, β_std_hist
end

# Set datatype
dType = Float32

# Make network(s)
nNeurons = [784, 32, 32, 10]
nLayers = length(nNeurons) - 1
activation = [relu, relu, identity]
dummyArray = [Dense(nNeurons[i], nNeurons[i+1], activation[i]) for i=1:nLayers] #Array of the Dense layers
Net0 = Chain(dummyArray...) # Splat the layers and Chain them
params(Net0)[1][:,:] .-= 0.025# .-= 0.03 #-= 0.04*abs.(randn(64, 784))
#params(Net0)[1][:,:] .*=-1*sign.(params(Net0)[1]) # :( )No learning with all negative weights
#params(Net0)[2][:,:] -= 2.5*ones(Float32, nNeurons[2])
Net_BP = deepcopy(Net0)
Net_hybrid = deepcopy(Net0)

# Hyper parameters
nEpochs = 5
batchsize = 64
ηAdam = 0.0001
ηSGD = 0.5
## Optimizer
#opt = Descent(ηSGD) #SGD
opt = ADAM(ηAdam)

# Data
trainsamples = 60000
testsamples = 10000

#LinearAlgebra.BLAS.set_num_threads(8)
#nT = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
#println(nT)


# Train hybrid network
# println("Training in hybrid-mode")
Random.seed!(33)
Net_hybrid, β_av_hist, β_std_hist = train_hybrid(Net_hybrid, batchsize, opt, nEpochs, trainsamples, testsamples)
println("\nTraining finished!\n")

# Train BP network
# println("Training in pure BP-mode")
# Random.seed!(33)
# train_BP(Net_BP, batchsize, opt, nEpochs, trainsamples, testsamples)




# data
# train_loader, test_loader = getdata(64, 60000, 10000)
# x = train_loader.data[1][:,1:5]
# y = train_loader.data[2][:,1:5]
# θ = params(Net_hybrid)
# z1_free = Net_hybrid[1](x)
# ∇θ = gradient(() -> mse(Net_hybrid(x), y), θ) # compute gradient

# z1_hat_best = similar(z1_free)
# grad = gradient(() -> mse(Net_hybrid[2:end](z1_free), y), params([z1_free]))
# ∇z1_free = grad[z1_free]

#=
Tasks:
1. Initial Steps ✓
1.1. Make NN ✓
1.2. Make predict function ✓
1.3. Make loss function ✓
1.4. Compute gradient with respect to x ✓
1.5. Compute gradient with respect to weights ✓

2. Intermediate Steps
2.1. Make lifted loss function for first layer ✓
2.2. Make FF loss function for subsequent layers ✓
2.3. Adjust FF functions to take z1 as input rather than x ✓
2.4. Derive optimal z via linearization of loss gradient (at this stage β is fixed) ✓
2.5. Implement computation of z1_hat ✓
2.6. Compute ∇θ for layer 1 ✓
2.7. make a basic BP training function ✓
2.8. Make Hybrid lifted/BP training implementation ✓

3. Advanced Steps
3.1. make function for finding β per batch ✓ (not really useful) ✓
3.2. make function for finding β per neuron per datapoint ✓
3.3. Find good choice of β ✓
One β per batch or one β per datapoint?... No! one beta per neuron per datapoint!
... Here β should be varied to recover gradients even for dead units
... and the first layer should be changed to a prototype layer.
... Prototype units
=#
