function get_loss(Net, w1x_plus_b1, z, activation)
    #TODO: Avoid unnecesary allocations by using inplace operations.
    # Preallocated dummy variables a and b might also be useful
    # First layer
    a = w1x_plus_b1
    b = a - z[1]
    J = 0.5*BLAS.dot(b,b)

    b = a - activation[1].(a)
    J -= 0.5*BLAS.dot(b,b)

    # Subsequent layers
    for i=2:Net.nLayers
        a = Net.w[i]*z[i-1] + Net.b[i]
        b = a - z[i]
        J += 0.5*BLAS.dot(b, b)
        b = a - activation[i].(a)
        J -= 0.5*BLAS.dot(b,b)
    end
    return J
end



# """Perform coordinate descent in z."""
# function run_BCD_LPOM!(nInnerIterations, x, z, y,
#                        W0, W1, b0, b1,
#                        a, b, denoms, w1, c,
#                        activation, high)
#     a .= W0*x .+ b0
#     c .= y .- b1 
#     b .= W1'*(c)


#     # Update all neurons in layer nPasses times
#     for pass=1:nInnerIterations
#         c = W1*z

#         # Iterate through the neurons
#         for i=1:length(z)
#             z_old = z[i]
#             w1 = view(W1, :, i)

#             # Compute u_j
#             u = 0.0
#             v = 0.0

#             for j=1:length(y)
#                 cc = c[j] + b1[j] # cc = [W1z+b1][j]
#                 if cc<0.0
#                     u -= w1[j]*cc
#                 elseif (cc-high)>0.0
#                     v -= (w1[j]*(cc - high))
#                 end
#             end
#             # Update c
#             c .-= w1.*z_old

#             # Update z
#             z[i] = activation((a[i] + b[i] - BLAS.dot(c, w1) - u - v)*denoms[i])

#             # Update C
#             c .+= w1.*z[i]
#         end
#     end
# end

"""Perform coordinate descent in z. This version does not take an argument x.
It assumes that the preactivation a has already been precomputed.
This is useful for the first layer."""
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

            # Compute u_j and v_j
            u = 0.0
            v = 0.0
            for j=1:length(y)
                cc = c[j] + b1[j] # cc = [W1z+b1][j]
                if cc<0.0
                    u -= w1[j]*cc
                elseif (cc-high)>0.0
                    v -= (w1[j]*(cc - high))
                end
            end

            # Update c
            c .-= w1.*z_old

            # Update z
            z[i] = activation((a[i] + b[i] - BLAS.dot(c, w1) - u - v)*denoms[i])

            # Update C
            c .+= w1.*z[i]
        end
    end
end

function run_LPOM_inference(x, w1x_plus_b1, z, denoms, Net, nOuterIterations, nInnerIterations, activation)
    for i=1:nOuterIterations
        # Backwards through the layers
        # On the first iteration i=1 we start infering at layer nLayers-1
        # On subsequent iterations i>1 we start at nLayers-2 since the last thing updated in the previous
        # iterations was layer nLayers -1.
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
    @inbounds for (layer, w) in enumerate(Net.w[2:end])#use view?
        @inbounds for i=1:Net.nNeurons[layer+1]
            denoms[layer][i] = 1/(1 + BLAS.dot(view(w, :, i), view(w, :, i)))
        end
    end
    return denoms
end


function get_∇!(∇w, ∇b, x, w1x_plus_b1, z, activation, Net)
    # First layer
    ∇b[1] = activation[1].(w1x_plus_b1) - z[1]
    ∇w[1] = ∇b[1]*x'
    # Subsequent layers
    for i=2:Net.nLayers
        ∇b[i] = activation[i].(Net.w[i]*z[i-1] + Net.b[i]) - z[i]
        ∇w[i] = ∇b[i]*z[i-1]'
    end
end


function predict(Net, x, y, batchsize, activation)
	  """Train an MLP. Training is parallel across datapoints."""

	  nNeurons = Net.nNeurons

	  nBatches = trunc(Int16, length(y) / batchsize)
	  nSamples = length(y) - (length(y)%batchsize)
	  #Allocate array for activations
	  z = [[zeros(Float16, N) for N in nNeurons[2:end]] for i=1:batchsize]

	  # And a variable for counting batch number
	  batchIndex::Int32 = 0
	  correct::Int32 = 0
    z1_saturated_down::dType = 0
    z1_saturated_up::dType = 0
    z2_saturated_down::dType = 0
    z2_saturated_up::dType = 0
    z3_saturated_down::dType = 0
    z3_saturated_up::dType = 0
	  # Loop through mini batches
	  for i = 1:batchsize:nSamples
		    batchIndex += 1
		    start = i
		    stop = start + batchsize - 1

		    xBatch = x[start:stop]
		    yBatch = y[start:stop]

        LinearAlgebra.BLAS.set_num_threads(1)
        Threads.@threads for n = 1:batchsize
            # Precompute  w1x and w1x+b1
			      w1x_plus_b1::Array{dType,1} = Net.w[1]*xBatch[n] + Net.b[1]
			      # Get activations
            forward!(z[n], Net, activation, w1x_plus_b1)
		    end
        LinearAlgebra.BLAS.set_num_threads(8)


		    # Correct equals true if prediction is correct. Otherwise equals false.
		    z_out = [zk[end] for zk in z]
		    correct += sum((argmax.(z_out).-1).==yBatch)

        z1_saturated_up += sum([count(i->(i==Net.highLim[1]), zk[1]) for zk in z])
        z1_saturated_down += sum([count(i->(i==0), zk[1]) for zk in z])
        z2_saturated_up += sum([count(i->(i==Net.highLim[2]), zk[2]) for zk in z])
        z2_saturated_down += sum([count(i->(i==0), zk[2]) for zk in z])
        z3_saturated_up += sum([count(i->(i==Net.highLim[3]), zk[3]) for zk in z])
        z3_saturated_down += sum([count(i->(i==0), zk[3]) for zk in z])
	  end

	  Av_accuracy = 100*correct/nSamples

    z1_saturated_up /= (nSamples*Net.nNeurons[2])
    z1_saturated_down /= (nSamples*Net.nNeurons[2])
    if Net.nLayers>1
        z2_saturated_up /= (nSamples*Net.nNeurons[3])
        z2_saturated_down /= (nSamples*Net.nNeurons[3])
    end
    if Net.nLayers>2
        z3_saturated_up /= (nSamples*Net.nNeurons[4])
        z3_saturated_down /= (nSamples*Net.nNeurons[4])
    end

	  return (Av_accuracy,
            z1_saturated_up, z1_saturated_down,
            z2_saturated_up, z2_saturated_down,
            z3_saturated_up, z3_saturated_down)
end


function train_LPOM_threads(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, nEpochs, η, nOuterIterations, nInnerIterations, activation, outpath)
	  """Train an MLP. Training is parallel across datapoints."""

	  L = Net.nLayers
	  nNeurons = Net.nNeurons

	  nBatches = trunc(Int16, length(yTrain) / batchsize)
	  nSamples = length(yTrain) - (length(yTrain)%batchsize)
	  #Allocate arrays to save training metrics to
    correctArray = zeros(Int32, batchsize)
	  J_batch = zeros(dType, batchsize)
	  z_batch = [[zeros(dType, N) for N in nNeurons[2:end]] for i=1:batchsize]
	  ∇w_batch = [[zeros(dType, (nNeurons[n+1], nNeurons[n])) for n = 1:Net.nLayers] for i=1:batchsize]
	  ∇b_batch = [[zeros(dType, (nNeurons[n+1])) for n = 1:Net.nLayers] for i=1:batchsize]
    ∇w = [zeros(dType, (nNeurons[n+1], nNeurons[n])) for n = 1:Net.nLayers]
	  ∇b = [zeros(dType, (nNeurons[n+1])) for n = 1:Net.nLayers]
    denoms = [zeros(dType, N) for N in Net.nNeurons[2:end-1]]
	  # Arrays for ADAM
    # M_w = [zeros(dType, (Net.nNeurons[i+1], Net.nNeurons[i])) for i = 1:Net.nLayers]
	  # M_b = [zeros(dType, (Net.nNeurons[i+1])) for i = 1:Net.nLayers]
	  # V_w = [zeros(dType, (Net.nNeurons[i+1], Net.nNeurons[i])) for i = 1:Net.nLayers]
	  # V_b = [zeros(dType, (Net.nNeurons[i+1])) for i = 1:Net.nLayers]

	  #Loop across epochs
	  for epoch = 1:nEpochs
		    t=0
		    t0 = time()
		    #Shuffle data
		    order = randperm(length(yTrain))
		    xTrain = xTrain[order]
		    yTrain = yTrain[order]

		    # And a variable for counting batch number
		    batchIndex::Int32 = 0
		    correct::Int32 = 0
		    J::dType = 0

		    # Loop through mini batches
		    for i = 1:batchsize:nSamples
			      batchIndex += 1
			      start = i
			      stop = start + batchsize - 1
			      xBatch = @view xTrain[start:stop]
			      yBatch = @view yTrain[start:stop]

            #= precompute denominator
            TODO: Check ifs list comprehension slower than explicit loops?
            This piece of code might be critical when working on CIFAR10,
            since W0 will be rather large. =#
            #denoms = [[1/(1 + sum(abs2,view(w, :, i))) for i=1:Net.nNeurons[layer+1]] for (layer, w) in enumerate(Net.w[2:end])]
            denoms = compute_denoms(denoms, Net)
            LinearAlgebra.BLAS.set_num_threads(1)
            Threads.@threads for n = 1:batchsize
                # Precompute w1x+b1
				        w1x_plus_b1 = Net.w[1]*xBatch[n] + Net.b[1]

                # Get activations
                forward!(z_batch[n], Net, activation, w1x_plus_b1);
                correctArray[n] = ((argmax(z_batch[n][end])-1)==yBatch[n])
							  z_batch[n][end] = [yBatch[n]+1==k for k=1:10]
                z_batch[n] = run_LPOM_inference(xBatch[n], w1x_plus_b1, z_batch[n], denoms, Net, nOuterIterations, nInnerIterations, activation)
                # Get Gradient and update weights
                get_∇!(∇w_batch[n], ∇b_batch[n], xBatch[n], w1x_plus_b1, z_batch[n], activation, Net)

                # Get loss
                J_batch[n] = get_loss(Net, w1x_plus_b1, z_batch[n], activation)
			      end
            LinearAlgebra.BLAS.set_num_threads(8)

            ∇w = meanOfGrad(∇w, ∇w_batch, batchsize)
            ∇b = meanOfGrad(∇b, ∇b_batch, batchsize)

			      J += sum(J_batch)
			      # Correct equals true if prediction is correct. Otherwise equals false.
			      # z_out = [zk[end] for zk in z_batch]
			      correct += sum(correctArray)

			      Net.w -= η * ∇w
			      Net.b -= η * ∇b
			      # update_weights_ADAM(Net, η, ∇w, ∇b, M_w, M_b, V_w, V_b)

			      #print("\r$(@sprintf("%.2f", 100*batchIndex/nBatches))% complete")

		    end

		    acc_train = 100*correct/nSamples
		    Av_J = J/nSamples

        # fb = Net.get_fb(Net)
		    acc_test, z1_saturated_up, z1_saturated_down, z2_saturated_up, z2_saturated_down, z3_saturated_up, z3_saturated_down = predict(Net, xTest, yTest, testBatchsize, activation)
		    push!(Net.History["acc_train"], acc_train)
		    push!(Net.History["acc_test"], acc_test)
		    push!(Net.History["J"], Av_J)
        push!(Net.History["z1_sat_up"], z1_saturated_up)
        push!(Net.History["z1_sat_down"], z1_saturated_down)
        push!(Net.History["z2_sat_up"], z2_saturated_up)
        push!(Net.History["z2_sat_down"], z2_saturated_down)
        push!(Net.History["z3_sat_up"], z3_saturated_up)
        push!(Net.History["z3_sat_down"], z3_saturated_down)

		    t1 = time()
		    println("\nav. metrics: \tJ: $(@sprintf("%.8f", Av_J))\tacc_train: $(@sprintf("%.2f", acc_train))%\tacc_test: $(@sprintf("%.2f", acc_test))% \tProc. time: $(@sprintf("%.2f", t1-t0))\n")

		    push!(Net.History["runTimes"], t1-t0)
		    FileIO.save("$outpath", "Net", Net)
	  end
    return
end
