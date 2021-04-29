using FileIO; # Save and load network dictionaries
using Printf # For formatting numeric output
using Random; Random.seed!(32); rng = MersenneTwister(13)
using LinearAlgebra

"""Perform coordinate descent in z."""
function run_BCD_LPOM!(nInnerIterations, z, y,
                       W0, W1, b0, b1,
                       a, b, denoms, w1, c,
                       activation, high)
    c .= y .- b1 
    b .= W1'*(c)

    # Update all neurons in layer nPasses times
    for pass=1:nInnerIterations
        c = W1*z

        # Iterate through the neurons
        for i=1:length(z)
            z_old = z[i]
            w1 = view(W1, :, i)

            u = 0.0
            v = 0.0
            for j=1:length(y)
                cc = c[j] + b1[j]
                if cc<0.0
                    u -= w1[j]*cc
                elseif (cc-high)>0.0
                    v -= (w1[j]*(cc - high))
                end
            end

            c .-= w1.*z_old
            z[i] = activation((a[i] + b[i] - BLAS.dot(c, w1) - u - v)*denoms[i])
            c .+= w1.*z[i]
        end
    end
end

function run_LPOM_inference!(x, w1x_plus_b1, z, denoms, Net, nOuterIterations, nInnerIterations, activation)
    for i=1:nOuterIterations
        #= Iterate backwards through the layers. On the first iteration i=1 we start infering
        at layer nLayers-1. On subsequent iterations i>1 we start at nLayers-2 since the last
        thing updated in the previous iteration was layer nLayers -1. =#
        startLayer = Net.nLayers-1-(i>1) 
        for k=startLayer:-1:2
            a = Net.w[k]*z[k-1] + Net.b[k]
            run_BCD_LPOM!(nInnerIterations, z[k], z[k+1],
                          Net.w[k], Net.w[k+1], Net.b[k], Net.b[k+1],
                          a, similar(z[k]), denoms[k], similar(z[k+1]), similar(z[k+1]),
                          activation[k], Net.highLim[k])
        end
        # First layer
        run_BCD_LPOM!(nInnerIterations, z[1], z[2],
                      Net.w[1], Net.w[2], Net.b[1], Net.b[2],
                      w1x_plus_b1, similar(z[1]), denoms[1], similar(z[2]), similar(z[2]),
                      activation[1], Net.highLim[1])
        # Forwards through layers
        for k=2:Net.nLayers-1
            a = Net.w[k]*z[k-1] + Net.b[k]
            run_BCD_LPOM!(nInnerIterations, z[k], z[k+1],
                          Net.w[k], Net.w[k+1], Net.b[k], Net.b[k+1],
                          a, similar(z[k]), denoms[k], similar(z[k+1]), similar(z[k+1]),
                          activation[k], Net.highLim[k])
        end
    end
    return z
end

function compute_denoms(denoms, Net)
    for (layer, w) in enumerate(Net.w[2:end])
        for i=1:Net.nNeurons[layer+1]
            denoms[layer][i] = 1/(1 + BLAS.dot(view(w, :, i), view(w, :, i)))
        end
    end
    # One-line alternative.
    # denoms = [[1/(1+BLAS.dot(wcol, wcol)) for wcol in eachcol(w)] for w in Net.w[2:end]]
    return denoms
end

function get_∇_V2!(∇w, ∇b, ∇bDummy, X, Z, batchsize, activation, Net)
    # First layer
    ∇bDummy[1] = (activation[1].(Net.w[1]*X .+ Net.b[1]) - Z[1])/batchsize
    ∇w[1] = ∇bDummy[1]*X'
    ∇b[1] = reshape(sum(∇bDummy[1], dims=2), size(∇b[1]))#/batchsize # reshaping to drop singleton dimension
    # Subsequent layers
    for i=2:Net.nLayers
        ∇bDummy[i] = (activation[i].(Net.w[i]*Z[i-1] .+ Net.b[i]) - Z[i])/batchsize
        ∇w[i] = ∇bDummy[i]*Z[i-1]'
        ∇b[i] = reshape(sum(∇bDummy[i], dims=2), size(∇b[i]))#/batchsize
    end
end

function get_loss_V2(Net, W1X_plus_b1, X, Z, activation)
    #TODO: Avoid unnecesary allocations by using inplace operations.
    # Preallocated dummy variables a and b might also be useful
    # First layer
    # a = Net.w[1]*X .+ Net.b[1]
    b = W1X_plus_b1 - Z[1]
    J = 0.5*sum(abs2, b)
    
    b = W1X_plus_b1 - activation[1].(W1X_plus_b1)
    J -= 0.5*sum(abs2, b)
    
    # Subsequent layers
    for i=2:Net.nLayers
        a = Net.w[i]*Z[i-1] .+ Net.b[i]
        b = a - Z[i]
        J += 0.5*sum(abs2, b)
        b = a - activation[i].(a)
        J -= 0.5*sum(abs2,b)
    end
    return J
end

function loss_and_accuracy(dataloader, batchsize, Net, activation)
    acc = 0.0
    J = 0.0
    nsamples = dataloader.nobs
    Z = [zeros(Float32, N, batchsize) for N in Net.nNeurons[2:end]]
    for (X, Y) in dataloader
        W1X_plus_b1 = Net.w[1]*X.+Net.b[1]
        forward!(Z, Net, activation, W1X_plus_b1)
        acc += sum(argmax.(eachcol(Z[end])).==argmax.(eachcol(Y)))
        J += get_loss_V2(Net, W1X_plus_b1, X, Z, activation)
    end
    return (100*acc/nsamples, J/nsamples)
end

function train_LPOM_threads_V2(Net, xTrain, yTrain, xTest, yTest, batchsize, test_batchsize, nEpochs, η, nOuterIterations, nInnerIterations, activation, outpath, numThreads)
	  """Train an MLP. Training is parallel across datapoints."""
    LinearAlgebra.BLAS.set_num_threads(numThreads)

    Z = [zeros(Float32, N, batchsize) for N in Net.nNeurons[2:end]]
    W1X_plus_b1w = [zeros(Float32, (Net.nNeurons[n+1], Net.nNeurons[n])) for n = 1:Net.nLayers]
    ∇w = [zeros(Float32, (Net.nNeurons[n+1], Net.nNeurons[n])) for n = 1:Net.nLayers]
	  ∇b = [zeros(Float32, (Net.nNeurons[n+1])) for n = 1:Net.nLayers]
    ∇bDummy = [zeros(Float32, (Net.nNeurons[n+1], batchsize)) for n = 1:Net.nLayers]
    denoms = [zeros(Float32, N) for N in Net.nNeurons[2:end-1]]

    trainloader, testloader = get_data(batchsize, test_batchsize)
    nSamples = trainloader.nobs

	  #Loop across epochs
	  for epoch = 1:nEpochs
		    t=0
		    t0 = time()

		    # And a variable for counting batch number
		    correct::Int32 = 0
		    J::Float32 = 0

		    # Loop through mini batches
        for (X, Y) in trainloader
            # Precompute
            W1X_plus_b1 = Net.w[1]*X.+Net.b[1]
            denoms = compute_denoms(denoms, Net)

            # FF predictions and clamping
            forward!(Z, Net, activation, W1X_plus_b1)
            correct += sum(argmax.(eachcol(Z[end])).==argmax.(eachcol(Y)))
            Z[end] .= Y

            #= Process individual datapoints: Note that BLAS should be single threaded
            in the threaded loop in order to not obstruct the benefits of dataparallelism=#
            LinearAlgebra.BLAS.set_num_threads(1)
            Threads.@threads for n = 1:size(Y)[2] # Size of last batch might differ so we use size(Y)[2]
                zz = [view(Zi, :, n) for Zi in Z]
                run_LPOM_inference!(view(X,:,n), view(W1X_plus_b1, :, n), zz, denoms,
                                    Net, nOuterIterations, nInnerIterations, activation)
			      end
            LinearAlgebra.BLAS.set_num_threads(numThreads) # Now BLAS should be multithreaded again

            # update number of correct predictions and the loss
            J += get_loss_V2(Net, W1X_plus_b1, X, Z, activation)

            # Compute the gradient and update the weights
            get_∇_V2!(∇w, ∇b, ∇bDummy, X, Z, batchsize, activation, Net)
			      Net.w -= η * ∇w
			      Net.b -= η * ∇b

		    end

		    acc_train = 100*correct/nSamples
		    Av_J = J/nSamples

        acc_test, J_test = loss_and_accuracy(testloader, test_batchsize, Net, activation)
		    push!(Net.History["acc_train"], acc_train)
		    push!(Net.History["acc_test"], acc_test)
		    push!(Net.History["J"], Av_J)

		    t1 = time()
		    println("\nav. metrics: \tJ: $(@sprintf("%.8f", Av_J))\tacc_train: $(@sprintf("%.2f", acc_train))%\tacc_test: $(@sprintf("%.2f", acc_test))% \tProc. time: $(@sprintf("%.2f", t1-t0))\n")

		    push!(Net.History["runTimes"], t1-t0)
		    FileIO.save("$outpath", "Net", Net)
	  end
    return
end

function get_data(batchsize, test_batchsize)

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
    test_loader = DataLoader((xtest, ytest), batchsize=test_batchsize)

    return train_loader, test_loader
end
