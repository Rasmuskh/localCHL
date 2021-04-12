using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Flux.Losses: mse
using MLDatasets
using LinearAlgebra
using Statistics
using Random; Random.seed!(3323); rng = MersenneTwister(12333)
using OhMyREPL
include("plottingFunctions.jl")
include("proto_layers.jl")
include("LoadData.jl")

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

    θ = Flux.params(Net)

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
        #plot_filters(Flux.params(Net[1])[1]', 8, 8, 800, 800, 28, 28, "/home/rasmus/Documents/localCHL/output/resurectionNet/FiltersBP/epoch$(epoch).png")
        t2 = time()
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
        println("  Runtime: $(t2-t1)")
        println("mean(θ[2]): $(mean(θ[2]))")
    end
    return Net
end

function Energy_of_neuron(di, z1i, ri, α, γ)
    E = 0.5*(1-γ)^2*ri^2*(z1i-1/γ)^2 + 0.5*α*HS(di-ri+(1-γ)*ri*z1i)
    return E
end

function  hybrid_grad_proto(Net_hybrid, X, z1_free, ∇z1_free, V, r, activation)
    #TODO: Reduce number of allocations
    # batchsize = (size(X)[2])
    # numIn = size(X)[1]
    # numOut = size(V)[2]
    # D = reshape(sqrt.(sum(abs2, V.-reshape(X, (numIn, 1, batchsize)), dims=1)), (numOut, batchsize))
    # d = [di for di in eachcol(D)]
    # γ, α = Net_hybrid[1].γ, Net_hybrid[1].α
    # a = ((γ^(-1) .+ α*(1-γ)*r.*(r .- d))./(1 .+ (α*(1-γ)^2*r.^2)))

    batchsize = (size(x)[2])
    numIn = size(x)[1]
    numOut = size(V)[2]
    D = reshape(sqrt.(sum(abs2, V.-reshape(x, (numIn, 1, batchsize)), dims=1)), (numOut, batchsize))
    d = [di for di in eachcol(D)]
    γ, α = Net_hybrid[1].γ, Net_hybrid[1].α
    A = ((γ^(-1) .+ α*(1-γ)*r.*(r .- D))./(1 .+ (α*(1-γ)^2*r.^2)))
    a = [ai for ai in eachcol(A)]
    v = [vi for vi in eachcol(V)]

    x_batch = [col for col in eachcol(X)]
    z1_free_batch = [col for col in eachcol(z1_free)]
    ∇z1_free_batch = [col for col in eachcol(∇z1_free)]
    z1_hat_batch = [similar(r) for k=1:batchsize]
    ∇V = similar(V)
    ∇r = similar(r)
    ∇V_batch = [similar(V) for k=1:batchsize]
    ∇r_batch = [similar(r) for k=1:batchsize]

    β_best_Arr = [similar(r) for k=1:batchsize]
    LinearAlgebra.BLAS.set_num_threads(1)
    # Loop over datapoints in parallel
    Threads.@threads for i=1:batchsize
        # Loop over neurons
        for n=1:length(r)
            β = 1.0;  β_best_Arr[i][n] = β
            z1_hat_batch[i][n] = activation(a[i][n] - β*∇z1_free_batch[i][n])
            E1_hat = Energy_of_neuron(d[i][n], z1_hat_batch[i][n], r[n], α, γ)
            E1_free = Energy_of_neuron(d[i][n], z1_free_batch[i][n], r[n], α, γ)
            ΔE_max = (E1_hat - E1_free)/β 
            E1_hat_max = E1_hat
            #Choose β to maximize ΔE = E_hat - E_free
            #TODO: Optimize this section
            for t=1:10
                β *= 2
                zdummy = activation(a[i][n] - β*∇z1_free_batch[i][n])
                E1_hat = Energy_of_neuron(d[i][n], zdummy, r[n], α, γ)
                ΔE = (E1_hat - E1_free)/β
                if ΔE>=ΔE_max
                    ΔE_max = ΔE
                    z1_hat_batch[i][n] = zdummy
                    β_best_Arr[i][n] = β
                else
                    break
                end

            end
            #∇r_batch[i][n] =  (z1_free_batch[i][n] .- z1_hat_batch[i][n])/β_best_Arr[i][n]  # !!!
            ∇V_batch[i][:,n] = -(x_batch[i]-v[n])/d[i][n]*α*(1-γ)*r[n]*(HS(d[i][n]-r[n] + (1-γ)*r[n]*z1_hat_batch[i][n])^2 - HS(d[i][n]-r[n] + (1-γ)*r[n]*z1_free_batch[i][n])^2)
        end
     end
    LinearAlgebra.BLAS.set_num_threads(8)
    β_av = sum(reduce(hcat,β_best_Arr))/length(r)
    β_std = std(reduce(hcat,β_best_Arr))/length(r)
    ∇V = mean(∇V_batch)
    #∇b1 = mean(∇b1_batch)  # !!!

    # # For debugging
    # # β = 4096.0; β_av = 0; β_std = 0
    # # z1_hat = get_z1_hat(X, W1, b1, β, ∇z1_free, activation)
    # # ∇b1, ∇W1 = get_∇θ1(X, z1_hat, z1_free, W1, b1, β, batchsize)

    # return (∇b1, ∇W1, β_av, β_std)
    return (∇V, β_av, β_std)
end



function train_hybrid(Net, batchsize, opt, nEpochs,  trainsamples, testsamples)

    train_loader, test_loader = getdata(batchsize, trainsamples, testsamples)

    θ = Flux.params(Net)
    activation = HS
    β_av_hist = zeros(dType, nEpochs)
    β_std_hist = zeros(dType, nEpochs)
    # loop over epochs
    for epoch in 1:nEpochs
        t1 = time()
        β_av = 0.0
        β_std = 0.0
        # loop over batches
        for (x, y) in train_loader

            ∇θ = gradient(() -> mse(Net(x), y), θ) # compute gradient

            # Get z1
            z1_free = Net[1](x)
            # get gradient wrt z1
            grad = gradient(() -> mse(Net[2:end](z1_free), y), Flux.params([z1_free]))
            ∇z1_free = grad[z1_free]
            

            # compute lifted learning signal and statistics about β
            #∇b1, ∇W1, β_av_batch, β_std_batch = hybrid_grad_proto(Net_hybrid, x, z1_free, ∇z1_free, θ[1], θ[2], activation)
            ∇V, β_av_batch, β_std_batch = hybrid_grad_proto(Net_hybrid, x, z1_free, ∇z1_free, θ[1], θ[2], activation)
            β_av += β_av_batch
            β_std += β_std_batch

            # replace first layer BP gradients with lifted gradients.
            ∇θ[θ[1]][:,:] = ∇V
            # ∇θ[θ[2]][:] = ∇b1

            # update parameters
            Flux.Optimise.update!(opt, θ, ∇θ)
        end
        β_av_hist[epoch] = β_av/trainsamples # get average batch mean(β)
        β_std_hist[epoch] = β_std/trainsamples # get average batch std(β)
        # Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, Net)
        test_loss, test_acc = loss_and_accuracy(test_loader, Net)
        #plot_filters(Flux.params(Net[1])[1], 8, 8, 800, 800, 28, 28, "/home/rasmus/Documents/localCHL/output/resurectionNet/FiltersResurection/epoch$(epoch).png")
        t2 = time()
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
        println("  β_av = $(β_av_hist[epoch]), β_std = $(β_std_hist[epoch])")
        println("  Runtime: $(t2-t1)")

    end
    return Net, β_av_hist, β_std_hist
end

# dataset = "MNIST"
# (xTrain, yTrain, xTest, yTest) = loadData(dataset, 5000, 5000)

# Set datatype
dType = Float32

# Make network(s)
nNeurons = [784, 16, 17, 10]
nLayers = length(nNeurons) - 1
activation = [relu, relu, identity]

# Net0 = Chain(proto(nNeurons[1], nNeurons[2], activation[1]),
#              Dense(nNeurons[2], nNeurons[3], activation[2]),
#              Dense(nNeurons[3], nNeurons[4], activation[3]))

Net0 = Chain(proto(nNeurons[1], nNeurons[2]),
             Dense(nNeurons[2], nNeurons[3], activation[2]),
             Dense(nNeurons[3], nNeurons[4], activation[3]))

# Switch to adversarial initial weights
#Flux.params(Net0)[1][:,:] .-= 0.03# .-= 0.03 #-= 0.04*abs.(randn(64, 784))
#Flux.params(Net0)[1][:,:] .*=-1*sign.(params(Net0)[1]) # :( No learning with all negative weights!
#params(Net0)[2][:,:] -= 2.5*ones(Float32, nNeurons[2])

Net_BP = deepcopy(Net0)
Net_hybrid = deepcopy(Net0)

# Hyper parameters
nEpochs = 3
batchsize = 16
ηAdam = 0.0005
ηSGD = 0.1
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
println("Training in hybrid-mode")
Random.seed!(33)
ps = Flux.params(Net_hybrid)
#@time k_means!(ps, 20)
@time Net_hybrid, β_av_hist, β_std_hist = train_hybrid(Net_hybrid, batchsize, opt, nEpochs, trainsamples, testsamples)
println("\nTraining finished!\n")

# Train BP network
# println("Training in pure BP-mode")
# Random.seed!(33)
# ps = Flux.params(Net_BP)
# @time k_means!(ps, 20)
# @time train_BP(Net_BP, batchsize, opt, nEpochs, trainsamples, testsamples)
# println("\nTraining finished!\n")



train_loader, test_loader = getdata(64, 60000, 10000);
x = train_loader.data[1][:,1:5];
y = train_loader.data[2][:,1:5];
V = Flux.params(Net_hybrid)[1];
r = Flux.params(Net_hybrid)[2];
println("...")
batchsize = (size(x)[2])
numIn = size(x)[1]
numOut = size(V)[2]
D = reshape(sqrt.(sum(abs2, V.-reshape(x, (numIn, 1, batchsize)), dims=1)), (numOut, batchsize))
γ, α = Net_hybrid[1].γ, Net_hybrid[1].α
A = ((γ^(-1) .+ α*(1-γ)*r.*(r .- D))./(1 .+ (α*(1-γ)^2*r.^2)))
a = [ai for ai in eachcol(D)]

println("...")

#=
Proto layer tasks:
1.1 Make basic FF-proto layer ✓
1.2 Get Kmeans initialization working ✓
1.3 Speed up prototype layer ✓
1.4 implement hybrid training
=#
