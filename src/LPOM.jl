

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

function run_LPOM_inference!(x, w1x_plus_b1, z, denoms, Net,
                             nOuterIterations, nInnerIterations, activation, highLim, nLayers)

    for i=1:nOuterIterations
        #= Iterate backwards through the layers. On the first iteration i=1 we start infering
        at layer nLayers-1. On subsequent iterations i>1 we start at nLayers-2 since the last
        thing updated in the previous iteration was layer nLayers -1. =#
        startLayer = nLayers-1-(i>1) 
        for k=startLayer:-1:2
            a = Net.w[k]*z[k-1] + Net.b[k]
            run_BCD_LPOM!(nInnerIterations, z[k], z[k+1],
                          Net.w[k], Net.w[k+1], Net.b[k], Net.b[k+1],
                          a, similar(z[k]), denoms[k], similar(z[k+1]), similar(z[k+1]),
                          activation[k], highLim[k])
        end
        # First layer
        run_BCD_LPOM!(nInnerIterations, z[1], z[2],
                      Net.w[1], Net.w[2], Net.b[1], Net.b[2],
                      w1x_plus_b1, similar(z[1]), denoms[1], similar(z[2]), similar(z[2]),
                      activation[1], highLim[1])
        # Forwards through layers
        for k=2:nLayers-1
            a = Net.w[k]*z[k-1] + Net.b[k]
            run_BCD_LPOM!(nInnerIterations, z[k], z[k+1],
                          Net.w[k], Net.w[k+1], Net.b[k], Net.b[k+1],
                          a, similar(z[k]), denoms[k], similar(z[k+1]), similar(z[k+1]),
                          activation[k], highLim[k])
        end
    end
    return z
end

function compute_denoms!(denoms, Net, nNeurons)

    for (layer, w) in enumerate(Net.w[2:end])
        for i=1:nNeurons[layer+1]
            denoms[layer][i] = 1/(1 + BLAS.dot(view(w, :, i), view(w, :, i)))
        end
    end
    # Slightly slower one-line alternative.
    # denoms = [[1/(1+BLAS.dot(wcol, wcol)) for wcol in eachcol(w)] for w in Net.w[2:end]]
    return denoms
end

function get_∇!(∇w, ∇b, ∇dummy, X, Z, batchsize, activation, Net, nLayers)

    # First layer
    ∇dummy[1] = (activation[1].(Net.w[1]*X .+ Net.b[1]) - Z[1])/batchsize
    ∇w[1] = ∇dummy[1]*X'
    ∇b[1] = reshape(sum(∇dummy[1], dims=2), size(∇b[1])) # reshaping to drop singleton dimension
    # Subsequent layers
    for i=2:nLayers
        ∇dummy[i] = (activation[i].(Net.w[i]*Z[i-1] .+ Net.b[i]) - Z[i])/batchsize
        ∇w[i] = ∇dummy[i]*Z[i-1]'
        ∇b[i] = reshape(sum(∇dummy[i], dims=2), size(∇b[i])) # reshaping to drop singleton dimension
    end
end

function get_loss(Net, nLayers, W1X_plus_b1, X, Z, activation)

    b = W1X_plus_b1 - Z[1]
    J = 0.5*sum(abs2, b)
    b = W1X_plus_b1 - activation[1].(W1X_plus_b1)
    J -= 0.5*sum(abs2, b)
    # Subsequent layers
    for i=2:nLayers
        a = Net.w[i]*Z[i-1] .+ Net.b[i]
        b = a - Z[i]
        J += 0.5*sum(abs2, b)
        b = a - activation[i].(a)
        J -= 0.5*sum(abs2,b)
    end
    return J
end

function get_accuracy(dataloader, batchsize, Net, nNeurons, activation)

    acc = 0.0
    nsamples = dataloader.nobs
    Z = [zeros(Float32, N, batchsize) for N in nNeurons[2:end]]
    for (X, Y) in dataloader
        W1X_plus_b1 = Net.w[1]*X.+Net.b[1]
        forward!(Z, Net, activation, W1X_plus_b1)
        acc += sum(argmax.(eachcol(Z[end])).==argmax.(eachcol(Y)))
    end

    return 100*acc/nsamples
end

"""Train an MLP using local contrastive Hebbian learning."""
function train_LPOM_threads(Net, args)

    LinearAlgebra.BLAS.set_num_threads(args.numThreads)
    nNeurons = args.nNeurons
    nLayers = args.nLayers

    Z = [zeros(Float32, N, args.batchsize) for N in nNeurons[2:end]]
    ∇w = [zeros(Float32, (nNeurons[n+1], nNeurons[n])) for n = 1:nLayers]
	  ∇b = [zeros(Float32, (nNeurons[n+1])) for n = 1:nLayers]
    ∇dummy = [zeros(Float32, (nNeurons[n+1], args.batchsize)) for n = 1:nLayers]
    denoms = [zeros(Float32, N) for N in nNeurons[2:end-1]]

    trainloader, testloader = get_data(args.batchsize, args.test_batchsize)
    nSamples = trainloader.imax

	  #Loop across epochs
	  for epoch = 1:args.nEpochs
		    t0 = time()

		    correct = 0
		    J_train = 0.0
        acc_train = 0.0
        acc_test = 0.0

		    # Loop through mini batches
        for (X, Y) in trainloader
            # Precompute
            W1X_plus_b1 = Net.w[1]*X .+ Net.b[1]
            compute_denoms!(denoms, Net, nNeurons)

            # Make FF predictions and clamp output units
            forward!(Z, Net, args.activation, W1X_plus_b1)
            correct += sum(argmax.(eachcol(Z[end])).==argmax.(eachcol(Y)))
            Z[end] .= Y

            #= Process individual datapoints: Note that BLAS should be single threaded
            in the threaded loop in order to not obstruct the benefits of dataparallelism=#
            LinearAlgebra.BLAS.set_num_threads(1)
            # Size of final batch might be less than batchsize so we use size(Y)[2]
            Threads.@threads for n = 1:size(Y)[2] 
                # use view to reference the columns of Z. Each column corresponds to a datapoint
                z = [view(zi, :, n) for zi in Z]
                run_LPOM_inference!(X[:, n], W1X_plus_b1[:, n], z, denoms,
                                    Net, args.nOuterIterations, args.nInnerIterations,
                                    args.activation, args.highLim, nLayers)
			      end
            # Now BLAS should be multithreaded again
            LinearAlgebra.BLAS.set_num_threads(args.numThreads) 

            # update number of correct predictions and the loss
            J_train += get_loss(Net, nLayers, W1X_plus_b1, X, Z, args.activation)

            # Compute the gradient and update the weights
            get_∇!(∇w, ∇b, ∇dummy, X, Z,  args.batchsize, args.activation, Net, nLayers)
			      Net.w -= args.η * ∇w
			      Net.b -= args.η * ∇b

		    end

		    acc_train = 100*correct/nSamples
		    J_train = J_train/nSamples

        acc_test = get_accuracy(testloader, args.test_batchsize, Net, nNeurons, args.activation)
		    push!(Net.History["acc_train"], acc_train)
		    push!(Net.History["acc_test"], acc_test)
		    push!(Net.History["J_train"], J_train)

		    t1 = time()
		    println("\nJ_train: $(@sprintf("%.8f", J_train))\tacc_train: $(@sprintf("%.2f", acc_train))%\tacc_test: $(@sprintf("%.2f", acc_test))% \tProc. time: $(@sprintf("%.2f", t1-t0))\n")
		    push!(Net.History["runTimes"], t1-t0)
		    FileIO.save("$(args.outpath)", "Net", Net)
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
    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true, partial=false)
    test_loader = DataLoader((xtest, ytest), batchsize=test_batchsize, partial=false)

    return train_loader, test_loader
end
