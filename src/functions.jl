using Statistics
using LinearAlgebra
using Printf # For formatting numeric output
using Random
using JLD2; # Save and load network dictionaries
using FileIO; # Save and load network dictionaries


mutable struct Net_CHL
    nNeurons::Array{Int64, 1}
	  nLayers::Int64
	  w::Array{Array{Float64, 2},1}
	  b::Array{Array{Float64, 1},1}
    highLim::Array{Float64,1}
	  num_updates::Int64
	  History::Dict{String, Array{Float64,1}}
end
Random.seed!(22)
rng = MersenneTwister(3)

function init_network(nNeurons, highLim)
	  """Initialize network weights and allocate arrays for training metric history."""
	  nLayers = length(nNeurons) - 1
	  # w = [sqrt(2 / nNeurons[i]) * randn(rng, Float64, (nNeurons[i+1], nNeurons[i])) for i = 1:nLayers]
    w = [1/sqrt(nNeurons[i] * nNeurons[i+1]) * randn(rng, Float64, (nNeurons[i+1], nNeurons[i])) for i = 1:nLayers]
	  b = [zeros(Float64, (nNeurons[i+1])) for i = 1:nLayers]

	  num_updates = 0
    # Network history dictionary
	  History = Dict(
	      "runTimes" => Float64[],
	      "acc_train" => Float64[],
	      "acc_test" => Float64[],
	      "J" => Float64[],
    )

	  # Network struct
	  Net = Net_CHL(nNeurons, nLayers, w, b, highLim, num_updates, History)
	  println("Network initialized")
	  return Net
end

"""ReLU function: Clamps input between 0 and infinity"""
function ReLU(PreActivation)
    return max.(0, PreActivation)
end

"""Hard sigmoid function: Clamps input between 0 and 1"""
function HardSigmoid(PreActivation)
    return min.(1, max.(0, PreActivation))
end

function Linear(PreActivation)
    return PreActivation
end

# function Clamp(z, low::Float64=0.0, high::Float64=Inf)
#     return min.(high, max.(low, z))
# end

"""Simple feed-forward pass"""
function forward!(z, Net, activation, w1x_plus_b1)
    z[1] = activation[1](w1x_plus_b1)

    @inbounds for i = 2:Net.nLayers-1
        z[i] =  activation[i](Net.w[i] * z[i-1] .+ Net.b[i])
    end

    #The final layer is linear
    L = Net.nLayers
    z[L] = activation[L](Net.w[L] * z[L-1] .+ Net.b[L])
end

# # ReLU implementations loss
# function get_loss(Net, w1x_plus_b1, z, activation)
#     #TODO: Avoid unnecesary allocations by using inplace operations.
#     # First layer
#     a = activation[1](w1x_plus_b1)
#     J = 0.5*BLAS.dot(z[1], z[1]) - z[1]'*(w1x_plus_b1) + 0.5*BLAS.dot(a, a)
#     # Subsequent layers
#     for i=2:Net.nLayers
#         a = Net.w[i]*z[i-1] + Net.b[i]
#         J += 0.5*BLAS.dot(z[i], z[i]) - z[i]'*a
#         a = activation[i](a)
#         J += 0.5*BLAS.dot(a, a)
#     end
#     return J
# end

function get_loss(Net, w1x_plus_b1, z, activation)
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
        a = Net.w[i]*z[i-1] + Net.b[i]
        b = a - z[i]
        J += 0.5*BLAS.dot(b, b)
        b = a - activation[i](a)
        J -= 0.5*BLAS.dot(b,b)
    end
    return J
end

function get_loss2(Net, w1x_plus_b1, z, activation)
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



"""Perform coordinate descent in z."""
function run_BCD_LPOM!(nInnerIterations, x, z, y,
                       W0, W1, b0, b1,
                       a, b, denoms, w1, c,
                       activation, high)
    a .= W0*x .+ b0
    c .= y .- b1 
    b = W1'*(c)


    # Update all neurons in layer nPasses times
    for pass=1:nInnerIterations
        c = W1*z

        # Iterate through the neurons
        for i=1:length(z)
            z_old = z[i]
            w1 = view(W1, :, i)

            # Compute u_j
            u = 0.0
            v = 0.0

            for j=1:length(y)
                cc = c[j] + b1[j] # cc = [W1z+b1][j]
                if cc<0.0
                    u -= w1[j]*cc
                elseif (cc-high)>0.0
                    v -= (w1[j]*cc - high)
                end
            end
            # Update c
            c .-= w1.*z_old

            # Update z
            z[i] = activation((a[i] + b[i] - BLAS.dot(c, w1) - u - v)*denoms[i])

            # Update C
            c .+= w1.*z[i]#z_new
        end
    end
end

"""Perform coordinate descent in z. This version does not take an argument x.
It assumes that the preactivation a has already been precomputed.
This is useful for the first layer."""
function run_BCD_LPOM!(nInnerIterations, z, y,
                       W0, W1, b0, b1,
                       a, b, denoms, w1, c,
                       activation, high)
    c .= y .- b1 
    b = W1'*(c)

    # Update all neurons in layer nPasses times
    for pass=1:nInnerIterations
        c = W1*z

        # Iterate through the neurons
        for i=1:length(z)
            z_old = z[i]
            w1 = view(W1, :, i)

            # Compute u_j
            u = 0.0
            v = 0.0
            for j=1:length(y)
                cc = c[j] + b1[j] # cc = [W1z+b1][j]
                if cc<0.0
                    u -= w1[j]*cc
                elseif (cc-high)>0.0
                    v -= (w1[j]*cc - high)
                end
            end
            # Update c
            c .-= w1.*z_old

            # Update z
            z[i] = activation((a[i] + b[i] - BLAS.dot(c, w1) - u - v)*denoms[i])

            # Update C
            c .+= w1.*z[i]#z_new
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
            run_BCD_LPOM!(nInnerIterations, z[k-1], z[k], z[k+1],
                          Net.w[k], Net.w[k+1], Net.b[k], Net.b[k+1],
                          similar(z[k]), similar(z[k]), denoms[k], similar(z[k+1]), similar(z[k+1]),
                          activation[k], Net.highLim[k])
        end
        # First layer
        run_BCD_LPOM!(nInnerIterations, z[1], z[2],
                      Net.w[1], Net.w[2], Net.b[1], Net.b[2],
                      w1x_plus_b1, similar(z[1]), denoms[1], similar(z[2]), similar(z[2]),
                      activation[1], Net.highLim[1])
        # Forwards through layers
        for k=2:Net.nLayers-1
            run_BCD_LPOM!(nInnerIterations, z[k-1], z[k], z[k+1],
                          Net.w[k], Net.w[k+1], Net.b[k], Net.b[k+1],
                          similar(z[k]), similar(z[k]), denoms[k], similar(z[k+1]), similar(z[k+1]),
                          activation[k], Net.highLim[k])
        end
    end
    return z
end

function meanOfGrad!(∇w, ∇w_batch, batchsize) 
    ∇w .= ∇w_batch[1]
    @inbounds for grad in ∇w_batch[2:end]
        @inbounds for (layer, gradLayer) in enumerate(grad)
            ∇w[layer] .+= gradLayer
        end
    end
    ∇w ./= batchsize
end


function get_∇!(∇w, ∇b, x, w1x_plus_b1, z, activation, Net)
    # First layer
    ∇b[1] = activation[1](w1x_plus_b1) - z[1]
    ∇w[1] = ∇b[1]*x'
    # Subsequent layers
    for i=2:Net.nLayers
        ∇b[i] = activation[i](Net.w[i]*z[i-1] + Net.b[i]) - z[i]
        ∇w[i] = ∇b[i]*z[i-1]'
    end
end

function predict(Net, x, y, batchsize, activation)
	  """Train an MLP. Training is parallel across datapoints."""

	  nNeurons = Net.nNeurons

	  nBatches = trunc(Int64, length(y) / batchsize)
	  nSamples = length(y) - (length(y)%batchsize)
	  #Allocate array for activations
	  z = [[zeros(Float64, N) for N in nNeurons[2:end]] for i=1:batchsize]

	  # And a variable for counting batch number
	  batchIndex::Int64 = 0
	  correct::Int64 = 0

	  # Loop through mini batches
	  for i = 1:batchsize:nSamples
		    batchIndex += 1
		    start = i
		    stop = start + batchsize - 1

		    xBatch = x[start:stop]
		    yBatch = y[start:stop]

        Threads.@threads for n = 1:batchsize
            # Precompute  w1x and w1x+b1
			      w1x_plus_b1::Array{Float64,1} = Net.w[1]*xBatch[n] + Net.b[1]
			      # Get activations
            forward!(z[n], Net, activation, w1x_plus_b1)
		    end

		    # Correct equals true if prediction is correct. Otherwise equals false.
		    z_out = [zk[end] for zk in z]
		    correct += sum((argmax.(z_out).-1).==yBatch)

	  end

	  Av_accuracy = 100*correct/nSamples

	  return Av_accuracy
end

function update_weights_ADAM(Net, η, ∇w, ∇b, M_w, M_b, V_w, V_b)
    Net.num_updates += 1
    β1 = 0.9
    β2 = 0.99
    ϵ = 10^(-8)

    ∇w2 = [∇.^2 for ∇ in ∇w]
    ∇b2 = [∇.^2 for ∇ in ∇b]

    # Get biased first and second moments
    M_w = β1*M_w + (1-β1)*∇w
    M_b = β1*M_b + (1-β1)*∇b
    V_w = β2*V_w + (1-β2)*∇w2
    V_b = β2*V_b + (1-β2)*∇b2
    # Get bias corrected moments
    M_w_hat = M_w/(1-β1^Net.num_updates)
    M_b_hat = M_b/(1-β1^Net.num_updates)
    V_w_hat = V_w/(1-β2^Net.num_updates)
    V_b_hat = V_b/(1-β2^Net.num_updates)

    # Compute the steps
    sqrt_V_w_hat = [sqrt.(dummy) for dummy in V_w_hat]
    step_w = [ M_w_hat[i]./(sqrt_V_w_hat[i].+ϵ) for i=1:length(M_w_hat)]

    sqrt_V_b_hat = [sqrt.(dummy) for dummy in V_b_hat]
    step_b = [ M_b_hat[i]./(sqrt_V_b_hat[i].+ϵ) for i=1:length(M_b_hat)]

    Net.w -= η * step_w
    Net.b -= η * step_b

end

function trainThreads(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, nEpochs, η, nOuterIterations, nInnerIterations, activation, outpath)
	  """Train an MLP. Training is parallel across datapoints."""

	  L = Net.nLayers
	  nNeurons = Net.nNeurons

	  nBatches = trunc(Int64, length(yTrain) / batchsize)
	  nSamples = length(yTrain) - (length(yTrain)%batchsize)
	  #Allocate arrays to save training metrics to
    correctArray = zeros(Int16, batchsize)
	  J_batch = zeros(Float64, batchsize)
	  z_batch = [[zeros(Float64, N) for N in nNeurons[2:end]] for i=1:batchsize]
	  ∇w_batch = [[zeros(Float64, (nNeurons[n+1], nNeurons[n])) for n = 1:Net.nLayers] for i=1:batchsize]
	  ∇b_batch = [[zeros(Float64, (nNeurons[n+1])) for n = 1:Net.nLayers] for i=1:batchsize]
    ∇w = [zeros(Float64, (nNeurons[n+1], nNeurons[n])) for n = 1:Net.nLayers]
	  ∇b = [zeros(Float64, (nNeurons[n+1])) for n = 1:Net.nLayers]

	  # Arrays for ADAM
    # M_w = [zeros(Float64, (Net.nNeurons[i+1], Net.nNeurons[i])) for i = 1:Net.nLayers]
	  # M_b = [zeros(Float64, (Net.nNeurons[i+1])) for i = 1:Net.nLayers]
	  # V_w = [zeros(Float64, (Net.nNeurons[i+1], Net.nNeurons[i])) for i = 1:Net.nLayers]
	  # V_b = [zeros(Float64, (Net.nNeurons[i+1])) for i = 1:Net.nLayers]

	  #Loop across epochs
	  for epoch = 1:nEpochs
		    t=0
		    t0 = time()
		    #Shuffle data
		    order = randperm(length(yTrain))
		    xTrain = xTrain[order]
		    yTrain = yTrain[order]

		    # And a variable for counting batch number
		    batchIndex::Int64 = 0
		    correct::Int64 = 0
		    J::Float64 = 0

		    # Loop through mini batches
		    for i = 1:batchsize:nSamples
			      batchIndex += 1
			      start = i
			      stop = start + batchsize - 1
			      xBatch = @view xTrain[start:stop]
			      yBatch = @view yTrain[start:stop]

            # precompute
            denoms = [[1/(1 + sum(abs2,view(w, :, i))) for i=1:Net.nNeurons[layer+1]] for (layer, w) in enumerate(Net.w[2:end])]
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
            meanOfGrad!(∇w, ∇w_batch, batchsize)
            meanOfGrad!(∇b, ∇b_batch, batchsize)

			      J += sum(J_batch)
			      # Correct equals true if prediction is correct. Otherwise equals false.
			      # z_out = [zk[end] for zk in z_batch]
			      correct += sum(correctArray)

			      Net.w -= η * ∇w
			      Net.b -= η * ∇b
			      # update_weights_ADAM(Net, η, ∇w, ∇b, M_w, M_b, V_w, V_b)

			      # print("\r$(@sprintf("%.2f", 100*batchIndex/nBatches))% complete")

		    end

		    acc_train = 100*correct/nSamples
		    Av_J = J/nSamples

        # fb = Net.get_fb(Net)
		    acc_test = predict(Net, xTest, yTest, testBatchsize, activation)
		    push!(Net.History["acc_train"], acc_train)
		    push!(Net.History["acc_test"], acc_test)
		    push!(Net.History["J"], Av_J)


		    t1 = time()
		    println("\nav. metrics: \tJ: $(@sprintf("%.8f", Av_J))\tacc_train: $(@sprintf("%.2f", acc_train))%\tacc_test: $(@sprintf("%.2f", acc_test))% \tProc. time: $(@sprintf("%.2f", t1-t0))")

		    push!(Net.History["runTimes"], t1-t0)
		    FileIO.save("$outpath", "Net", Net)
	  end
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
		    batchIndex::Int64 = 0
		    correct::Int64 = 0
		    J::Float64 = 0
        Z = [zeros(Float64, (N,batchsize)) for N in Net.nNeurons[2:end]];
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

