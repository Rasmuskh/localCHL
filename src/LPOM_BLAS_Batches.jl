
function get_lossBatch(Net, w1x_plus_b1, z, activation)
    #TODO: Avoid unnecesary allocations by using inplace operations.
    # Preallocated dummy variables a and b might also be useful
    # First layer
    a = w1x_plus_b1
    b = a - z[1]
    J = 0.5*BLAS.dot(b,b)

    b = a - activation[1](a)
    J -= 0.5*BLAS.dot(b,b)

    # Subsequent layers
    for i=2:Net.nLayers
        a = Net.w[i]*z[i-1] .+ Net.b[i]
        b = a - z[i]
        J += 0.5*BLAS.dot(b, b)
        b = a - activation[i](a)
        J -= 0.5*BLAS.dot(b,b)
    end
    return J
end

function run_LPOM_inferenceBatch(X, w1x_plus_b1, Z, denoms, Net, nOuterIterations, nInnerIterations, activation)
    for i=1:nOuterIterations
        # Backwards through the layers
        # On the first iteration i=1 we start infering at layer nLayers-1
        # On subsequent iterations i>1 we start at nLayers-2 since the last thing updated in the previous
        # iterations was layer nLayers -1.
        startLayer = Net.nLayers-1-(i>1) 
        for k=startLayer:-1:2
            run_BCD_LPOM_batch!(nInnerIterations, Z[k-1], Z[k], Z[k+1],
                           Net.w[k], Net.w[k+1], Net.b[k], Net.b[k+1],
                           similar(Z[k]), similar(Z[k]), denoms[k], similar(Z[k+1]), similar(Z[k+1]),
                          activation[k], Net.highLim[k],
                          similar(Z[k]), similar(Z[k]))
        end
        # First layer
        run_BCD_LPOM_batch!(nInnerIterations, X, Z[1], Z[2],
                      Net.w[1], Net.w[2], Net.b[1], Net.b[2],
                      similar(Z[1]), similar(Z[1]), denoms[1], similar(Z[2]), similar(Z[2]),
                      activation[1], Net.highLim[1],
                      similar(Z[1]), similar(Z[1]))
 
        # Forwards through layers
        for k=2:Net.nLayers-1
            run_BCD_LPOM_batch!(nInnerIterations, Z[k-1], Z[k], Z[k+1],
                          Net.w[k], Net.w[k+1], Net.b[k], Net.b[k+1],
                          similar(Z[k]), similar(Z[k]), denoms[k], similar(Z[k+1]), similar(Z[k+1]),
                          activation[k], Net.highLim[k],
                          similar(Z[k]), similar(Z[k]))
        end
    end
    return Z
end



"""Perform coordinate descent in z."""
function run_BCD_LPOM_batch!(nInnerIterations, X, Z, Y,
                       W0, W1, b0, b1,
                       a, b, denoms, w1, c,
                       activation, high,
                       U, V)
    a .= W0*X .+ b0
    c .= Y .- b1 
    b .= W1'*(c)
    batchsize = length(Z[1,:])

    # wTwZ = W1'*W1*Z
    # U = W1'*ReLU(-W1*Z .- b1)
    # V = W1'*ReLU(W1*Z .+ b1 .-1)

    # Update all neurons in layer nInnerIterations times
    for pass=1:nInnerIterations
        c = W1*Z # Does using "."  speed things up here?

        # Iterate through neurons
        for i=1:length(Z[:,1])
            Z_old = view(Z, i, :)

            w1 = view(W1, :, i)

            # wTwZ = w1'*c
            # U = W1'*ReLU(-c .- b1)
            # V = W1'*ReLU(c .+ b1 .-1)

            # #V = Net.w[k+1]'*ReLU(Net.w[k+1]*Z[k] .+ Net.b[k+1] .-1)

            # # compute u and v

            # subtract z_olds contribution from c
            # for col=1:batchsize
            #     c[:,col] .-= w1.*Z_old[col]
            # end
            # # compute z_new
            Z[i,:] .= denoms[i].*(view(a, i, :) .+ view(b, i, :))
            # # add z_news contribution from c
            # for col=1:batchsize
            #     c[:,col] .+= w1.*Z_old[col]
            # end
        end
    end

    # Update all neurons in layer nPasses times
    # for pass=1:nInnerIterations
    #     c = W1*z #update c

    #     # Iterate through the neurons
    #     for i=1:length(z)
    #         z_old = z[i]
    #         for j = 1:length(y)
    #             w1[j] = W1[j,i]#Check if rows and columns should be switched here
    #         end

    #         # Compute u_j
    #         u = 0.0
    #         v = 0.0
    #         for j=1:length(y)
    #             cc = c[j] + b1[j] # cc = [W1z+b1][j] at this point
    #             if cc<0.0
    #                 u -= w1[j]*cc
    #             elseif (cc-high)>0.0
    #                 v -= (w1[j]*cc - high)
    #             end
    #         end
    #         # Compute c
    #         for j = 1:length(y)
    #             c[j] -= w1[j]*z_old #update c
    #         end

    #         z_new = (a[i] + b[i] - BLAS.dot(c, w1) - u - v)/denoms[i]

    #         z_new = activation(z_new)
    #         for j = 1:length(y)
    #             c[j] += w1[j]*z_new #update c
    #         end
    #         z[i] = z_new
    #     end
    # end

    # Update all neurons in layer nPasses times
end


function trainBatches(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, nEpochs, η, nOuterIterations, nInnerIterations, activation, outpath)
	  #Loop across epochs
	  for epoch = 1:nEpochs
        # Restart clock
		    t=0; t0 = time()
		    #Shuffle data
		    order = randperm(length(yTrain))
		    xTrain = xTrain[order]
		    yTrain = yTrain[order]
        # If number of training datapoints is not divisible by batchsize,
        # then the remainders wont be used this epoch.g
        nSamples = length(yTrain) - (length(yTrain)%batchsize)

		    # And a variable for counting batch number
		    batchIndex::Int16 = 0
		    correct::Int16 = 0
		    J::Float16 = 0
        Z = [zeros(Float16, (N,batchsize)) for N in Net.nNeurons[2:end]];
		    # Loop through mini batches
		    for i = 1:batchsize:nSamples
            batchIndex += 1
			      start = i
			      stop = start + batchsize - 1
			      xBatch = @view xTrain[start:stop]
			      yBatch = @view yTrain[start:stop]

            X = reduce(hcat, xTrain[1:batchsize])
            # precompute
            denoms = [[1/(1 + sum(abs2,w[:,i])) for i=1:Net.nNeurons[layer+1]] for (layer, w) in enumerate(Net.w[2:end])]

            w1x_plus_b1 = Net.w[1]*X.+Net.b[1]
            forward!(Z, Net, activation, w1x_plus_b1);
            run_LPOM_inferenceBatch(X, w1x_plus_b1, Z, denoms, Net, nOuterIterations, nInnerIterations, activation)
            #infer activations
            #Compute gradient
            J = get_loss2(Net, w1x_plus_b1, Z, activation)

        end
    end
end

