using Flux, Statistics
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Flux.Losses: mse
using Base: @kwdef
using CUDA
using MLDatasets
using Random; Random.seed!(32); rng = MersenneTwister(13)

#=
This file is based on the Flux model zoo's MNIST MLP example:
https://github.com/FluxML/model-zoo/tree/master/vision/mlp_mnist
=#

#LinearAlgebra.BLAS.set_num_threads(8)

function getdata(batchsize, device)
    ENV["DATADEPS_ALWAYS_ACCEPT"] = "true"

    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
	
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

function build_model(; imgsize=(28,28,1), nclasses=10)
    return Chain(
 	      Dense(prod(imgsize), 512, relu),
        Dense(512, 512, relu),
        Dense(512, nclasses, identity))
end

function loss_and_accuracy(data_loader, Net, device)
    acc = 0
    ls = 0.0f0
    num = 0
    for (x, y) in data_loader
        x, y = device(x), device(y)
        ŷ = Net(x)
        # ls += logitcrossentropy(model(x), y, agg=sum)
        ls += mse(Net(x), y, agg=sum)
        acc += sum(onecold(cpu(Net(x))) .== onecold(cpu(y)))
        num +=  size(x, 2)
    end
    return ls / num, acc / num
end

function train_flux(Net, η, batchsize, nEpochs, use_cuda)

    if CUDA.functional() && use_cuda
        @info "Training on CUDA GPU"
        CUDA.allowscalar(false)
        device = gpu
    else
        @info "Training on CPU"
        device = cpu
    end

    # Create test and train dataloaders
    train_loader, test_loader = getdata(batchsize, device)

    # access trainable parameters
    ps = Flux.params(Net) # model's trainable parameters

    ## Optimizer
    opt = Descent(η) #SGD
    #opt = ADAM(args.η)

    ## Training
    for epoch in 1:nEpochs
        t1 = time()
        for (x, y) in train_loader
            x, y = device(x), device(y) # transfer data to device
            # gs = gradient(() -> logitcrossentropy(model(x), y), ps) # compute gradient
            gs = gradient(() -> mse(Net(x), y), ps) # compute gradient
            Flux.Optimise.update!(opt, ps, gs) # update parameters
        end

        # Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, Net, device)
        test_loss, test_acc = loss_and_accuracy(test_loader, Net, device)
        t2 = time()
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
        println("  Runtime: $(t2-t1)")
    end
    return Net
end

# Run training
Net = build_model()
Net = train_flux(Net, 0.1, 64, 5, false)
