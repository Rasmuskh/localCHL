using Flux
using Flux.Data: DataLoader
using Flux: onehotbatch, onecold, @epochs
using Flux.Losses: logitcrossentropy
using Flux.Losses: mse
using MLDatasets
using LinearAlgebra
using Statistics
using Random; Random.seed!(3323); rng = MersenneTwister(12333)


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

function lifted_E_layer1(x, z, W0, b0)
    a = W0*x .+ b0 .- z
    return BLAS.dot(a,a)
end

function lifted_loss_layer1(x, z_hat, z_free, W0, b0, β)
    E_hat = lifted_E_layer1(x, z_hat, W0, b0)
    E_free = lifted_E_layer1(x, z_free, W0, b0)
    return (E_hat - E_free)/β
end

# function loss(x, y)
#     pred = Net(x)
#     mse(y, pred)
#     #sum((target .- y).^2)
# end

# function loss_2nd_layer_onwards(z, y)
#     pred = Net[2:end](z)
#     mse(y, pred)
#     #sum((target .- y).^2)
# end



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

function train_BP(Net, η, batchsize, opt, nEpochs, trainsamples, testsamples)

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
        t2 = time()
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
        println("  Runtime: $(t2-t1)")
    end
    return Net
end

function maximize_gap_batch(x, ∇z1_free, W0, b0, β0, βmax, E_z1_free, activation)
    β = β0
    β_best = 0
    ΔE_best = 0
    #println(b0)
    z1_hat_best = similar(b0) #z1 has the same shape size as b0
    i = 0
    while β<βmax
        i += 1
        z1_hat = get_z1_hat(x, W0, b0, β, ∇z1_free, activation)
        E_z1_hat = lifted_E_layer1(x, z1_hat, W0, b0)
        ΔE = (E_z1_hat - E_z1_free)/β #  E_z1_free not really necessary here as it is constant!
        if (ΔE > ΔE_best) || (i==1)
            ΔE_best = ΔE
            β_best = β
            z1_hat_best = z1_hat
        #elseif ΔE<ΔE_best
        #    break
        end
        β *= 2
    end
    return (β_best, z1_hat_best)
end

# function maximize_gap_xi(X, ∇z1_free, W0, b0, β0, βmax, E_z1_free, activation)
#     xArray = [xi for xi in eachcol(X)]
#     for x in xArray
#         β = β0
#         β_best = 0
#         ΔE_best = 0
#         #println(b0)
#         z1_hat_best = similar(b0) #z1 has the same shape size as b0
#         i = 0
#         while β<βmax
#             i += 1
#             z1_hat = get_z1_hat(x, W0, b0, β, ∇z1_free, activation)
#             E_z1_hat = lifted_E_layer1(x, z1_hat, W0, b0)
#             ΔE = (E_z1_hat - E_z1_free)/β #  E_z1_free not really necessary here as it is constant!
#             if (ΔE > ΔE_best) || (i==1)
#                 ΔE_best = ΔE
#                 β_best = β
#                 z1_hat_best = z1_hat
#             elseif ΔE<ΔE_best
#                 break
#             end
#             β *= 2
#         #get z1_hat[i] by finiding optimal β
#         #get ∇θ[1:2] from β, z1_hat and z1_free

function get_z1_hat(x, W0, b0, β, ∇z1_free, activation)
    z1_hat = activation.(W0*x .+b0 .-β*∇z1_free)
    return z1_hat
end

function get_∇θ1(x, z1_hat, z1_free, W0, b0, β, batchsize)
    #println("z1_free", size(z1_free))
    #println("z1_hat", size(z1_hat))
    ∇b1 = (z1_free .- z1_hat)/β
    ∇W1 = (∇b1*x')

    ∇b1 = sum(∇b1, dims=2)#/batchsize
    #∇W1 /= batchsize
    return (∇b1, ∇W1)
end

function f(X, z1_free, ∇z1_free, W1, b1, activation)
    batchsize = (size(X)[2])
    x_batch = [col for col in eachcol(X)]
    z1_free_batch = [col for col in eachcol(z1_free)]
    ∇z1_free_batch = [col for col in eachcol(∇z1_free)]
    z1_hat_batch = [similar(b1) for k=1:batchsize]#similar(z1_free_batch)
    ∇W1 = similar(W1)
    ∇b1 = similar(b1)
    ∇W1_batch = [similar(W1) for k=1:batchsize]
    ∇b1_batch = [similar(b1) for k=1:batchsize]



    Threads.@threads for i=1:batchsize # Loop over datapoints
        a = W1*x_batch[i] .+ b1
        for n=1:length(b1) # Loop over neurons
            β = 0.1; β_best = β
            z1_hat_batch[i][n] = activation(a[n] - β*∇z1_free_batch[i][n])
            E1_hat = ((a[n] - z1_hat_batch[i][n])^2)
            E1_free = ((a[n] - z1_free_batch[i][n])^2)
            ΔE_max = (E1_hat - E1_free)/β
            #Choose β to maximize ΔE = E_hat - E_free
            for t=1:12
                z_dummy = activation(a[n] - β*∇z1_free_batch[i][n])
                E1_hat = ((a[n] - z1_hat_batch[i][n])^2)
                E1_free = ((a[n] - z1_free_batch[i][n])^2)
                ΔE = (E1_hat - E1_free)/β
                if ΔE>ΔE_max
                    ΔE_max = ΔE
                    z1_hat_batch[i][n] = z_dummy
                    β_best = β
                else
                    break
                end
                β *= 2
            end
            ∇b1_batch[i][n] =  (z1_free_batch[i][n] .- z1_hat_batch[i][n])/β
            ∇W1_batch[i][n,:] =  ∇b1_batch[i][n]*x_batch[i]'
        end
    end
    ∇W1 = mean(∇W1_batch)
    ∇b1 = mean(∇b1_batch)
    

    # β = 1.0
    # z1_hat = get_z1_hat(X, W1, b1, β, ∇z1_free, activation)
    # ∇b1, ∇W1 = get_∇θ1(X, z1_hat, z1_free, W1, b1, β, batchsize)

    return (∇b1, ∇W1)
end


function train_hybrid(Net, η, batchsize, opt, nEpochs,  trainsamples, testsamples)

    train_loader, test_loader = getdata(batchsize, trainsamples, testsamples)

    θ = params(Net)
    activation = relu

    # Training
    for epoch in 1:nEpochs
        t1 = time()
        for (x, y) in train_loader
            z1_free = Net[1](x)
            ∇θ = gradient(() -> mse(Net(x), y), θ) # compute gradient

            grad = gradient(() -> mse(Net[2:end](z1_free), y), params([z1_free]))
            ∇z1_free = grad[z1_free]
            ∇b1, ∇W1 = f(x, z1_free, ∇z1_free, θ[1], θ[2], activation)

            ∇θ[θ[1]][:,:] = ∇W1
            ∇θ[θ[2]][:] = ∇b1
            Flux.Optimise.update!(opt, θ, ∇θ) # update parameters
        end

        
        # Report on train and test
        train_loss, train_acc = loss_and_accuracy(train_loader, Net)
        test_loss, test_acc = loss_and_accuracy(test_loader, Net)
        t2 = time()
        println("Epoch=$epoch")
        println("  train_loss = $train_loss, train_accuracy = $train_acc")
        println("  test_loss = $test_loss, test_accuracy = $test_acc")
        println("  Runtime: $(t2-t1)")
    end
    return Net
end

# Make network(s)
nNeurons = [784, 64, 64 ,10]
nLayers = length(nNeurons) - 1
activation = [relu, relu, identity]
dummyArray = [Dense(nNeurons[i], nNeurons[i+1], activation[i]) for i=1:nLayers] #Array of the Dense layers
Net0 = Chain(dummyArray...) # Splat the layers and Chain them
params(Net0)[1][:,:] -= 0.02*abs.(randn(64, 784))
#params(Net0)[2][:,:] -= 2.5*ones(Float32, nNeurons[2])
Net_BP = deepcopy(Net0)
Net_hybrid = deepcopy(Net0)

# Hyper parameters
nEpochs = 2
batchsize = 64
η = 0.5
## Optimizer
opt = Descent(η) #SGD
#opt = ADAM(η)

# Data
trainsamples = 10000
testsamples = 10000

LinearAlgebra.BLAS.set_num_threads(8)
nT = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
println(nT)


# Train hybrid network
println("Training in hybrid-mode")
Random.seed!(3323)
train_hybrid(Net_hybrid, η, batchsize, opt, nEpochs, trainsamples, testsamples)
println("\nTraining finished!\n")

# Train BP network
println("Training in pure BP-mode")
Random.seed!(3323)
train_BP(Net_BP, η, batchsize, opt, nEpochs, trainsamples, testsamples)
println("\nTraining finished!\n")



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
3.1. make function for finding β per batch ✓ (not really useful)
3.2. make function for finding β per datapoint
3.3. Find good choice of β
One β per batch or one β per datapoint?... No! one beta per neuron per datapoint!
... Here β should be varied to recover gradients even for dead units
... and the first layer should be changed to a prototype layer.
... Prototype units
=#
