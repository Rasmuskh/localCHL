using Statistics
using LinearAlgebra
using Printf # For formatting numeric output
using Random
using JLD2; # Save and load network dictionaries
using FileIO; # Save and load network dictionaries


mutable struct Net_CHL
    nNeurons::Array{Int64, 1}
	  η::Float64
	  γ::Float64
	  nLayers::Int64
	  w::Array{Array{Float64, 2},1}
	  g::Array{Array{Float64, 2},1}
	  u::Array{Array{Float64, 2},1}
	  b::Array{Array{Float64, 1},1}
	  num_updates::Int64
	  History::Dict{String, Array{Float64,1}}
end
Random.seed!(471)
rng = MersenneTwister(135)

function init_network(nNeurons, γ, η)
	  """Initialize network weights and allocate arrays for training metric history."""
	  nLayers = length(nNeurons)
	  w = [sqrt(2 / nNeurons[i]) * randn(rng, Float64, (nNeurons[i+1], nNeurons[i])) for i = 1:nLayers-1]
	  g = [sqrt(2 / nNeurons[i]) * randn(rng, Float64, (nNeurons[i], nNeurons[i+1])) for i = 1:nLayers-1]
    u = [sqrt(2 / nNeurons[i]) * randn(rng, Float64, (nNeurons[i], nNeurons[end])) for i = 1:nLayers-1]
	  b = [zeros(Float64, (nNeurons[i+1])) for i = 1:nLayers-1]

	  num_updates = 0
    # Network history dictionary
	  History = Dict(
	      "runTimes" => Float64[],
	      "acc_train" => Float64[],
	      "acc_test" => Float64[],
	      "J" => Float64[],
	      "E_free" => Float64[],
	      "E_clamped" => Float64[],
        "av_z1_pos" => Float64[],
        "sum_z1" => Float64[],

    )
    
	  # Network struct
	  Net = Net_CHL(nNeurons, η, γ, nLayers, w, g, u, b, num_updates, History)
	  println("Network initialized")
	  return Net
end

function ReLU(PreActivation)
    """ReLU function: Clamps input between 0 and infinity"""
    return max.(0, PreActivation)
end

function energy(z, x, w1x_plus_b1, normW2, Net)
    E = 0.5 * (z[1] - 2*w1x_plus_b1)'*z[1] + normW2[1] * (x'*x) + Net.b[1]'*Net.b[1]
    for i = 2:Net.nLayers-1
        E += 0.5*Net.γ^(i-1)*z[i]'*(z[i] -2*(Net.w[i]*z[i-1]+Net.b[i]))
        E +=  0.5*Net.γ^(i-1) * normW2[i] * (z[i-1]'*z[i-1]) + Net.b[i]'*Net.b[i]
    end
    return E
end

function forward(z, Net, activation, w1x_plus_b1)
    """Simple forward pass"""
    z[1] = activation(w1x_plus_b1)
    @inbounds for i = 2:Net.nLayers-2
        z[i] =  activation(Net.w[i] * z[i-1] + Net.b[i])
    end
    #The final layer is linear
    L = Net.nLayers-1
    z[L] = Net.w[L] * z[L-1] + Net.b[L]
    return z
end


function get_z(w1x_plus_b1, normW2, wT, z, Net, activation, nIter, clamped)
        for u=1:nIter
            z[1] = activation((w1x_plus_b1 + Net.γ*wT[2]*z[2])/(1 + Net.γ*normW2[2]))
            @inbounds for i=2:Net.nLayers-2
                z[i] = activation((Net.w[i]*z[i-1] + Net.b[i] + Net.γ*wT[i+1]*z[i+1])
                                  /(1 + Net.γ*normW2[i+1]))
            end
            if clamped==false
                L = Net.nLayers - 1
                z[L] = Net.w[L]*z[L-1] + Net.b[L]
            end

        end
    return z
end

function get_z_direct_random_feedback(w1x_plus_b1, normW2, wT, z, Net, activation, nIter, clamped)
    for u=1:nIter
        z[1] = activation((w1x_plus_b1 + Net.γ*wT[2]*z[end])/(1 + Net.γ*normW2[2]))
        @inbounds for i=2:Net.nLayers-2
            z[i] = activation((Net.w[i]*z[i-1] + Net.b[i] + Net.γ*wT[i+1]*z[end])
                              /(1 + Net.γ*normW2[i+1]))
        end
        if clamped==false
            L = Net.nLayers - 1
            z[L] = Net.w[L]*z[L-1] + Net.b[L]
        end

    end
    return z
end

function get_z_fast2(w1x_plus_b1, normW2, wT, z_clamped, z_free, Net, activation, nIter)
    # run inference of clamped activations
    for u=1:nIter
        # Forwards
        z_clamped[1] = activation((w1x_plus_b1 + Net.γ*wT[2]*z_clamped[2])/(1 + Net.γ*normW2[2]))
        @inbounds for i=2:Net.nLayers-2
            z_clamped[i] = activation((Net.w[i]*z_clamped[i-1] + Net.b[i] + Net.γ*wT[i+1]*z_clamped[i+1])
                                      /(1 + Net.γ*normW2[i+1]))
        end
        # Backwards
        @inbounds for i=Net.nLayers-2:-1:2
            z_clamped[i] = activation((Net.w[i]*z_clamped[i-1] + Net.b[i] + Net.γ*wT[i+1]*z_clamped[i+1])
                                      /(1 + Net.γ*normW2[i+1]))
        end
        z_clamped[1] = activation((w1x_plus_b1 + Net.γ*wT[2]*z_clamped[2])/(1 + Net.γ*normW2[2]))
    end
    # Get psudo-free output
    L = Net.nLayers - 1
    z_free[L] = Net.w[L]*z_clamped[L-1] + Net.b[L]
    # get k'th pseudo-free activations via k-1 clamped activation and k+1 pseudo-free activation
    @inbounds for i in (Net.nLayers-2):-1:2
        z_free[i] = activation((Net.w[i]*z_clamped[i-1] + Net.b[i] + Net.γ*wT[i+1]*z_free[i+1])
                               /(1 + Net.γ*normW2[i+1]))
    end
    z_free[1] = activation((w1x_plus_b1 + Net.γ*wT[2]*z_free[2])/(1 + Net.γ*normW2[2]))
    return z_clamped, z_free
end

function get_z_fast(w1x_plus_b1, normW2, wT, z_clamped, z_free, Net, activation, nIter)
    # run inference of clamped activations
    for u=1:nIter
        z_clamped[1] = activation((w1x_plus_b1 + Net.γ*wT[2]*z_clamped[2])/(1 + Net.γ*normW2[2]))
        @inbounds for i=2:Net.nLayers-2
            z_clamped[i] = activation((Net.w[i]*z_clamped[i-1] + Net.b[i] + Net.γ*wT[i+1]*z_clamped[i+1])
                              /(1 + Net.γ*normW2[i+1]))
        end
    end
    # Get psudo-free output
    L = Net.nLayers - 1
    z_free[L] = Net.w[L]*z_clamped[L-1] + Net.b[L]
    # get k'th pseudo-free activations via k-1 clamped activation and k+1 pseudo-free activation
    @inbounds for i in (Net.nLayers-2):-1:2
        z_free[i] = activation((Net.w[i]*z_clamped[i-1] + Net.b[i] + Net.γ*wT[i+1]*z_free[i+1])
                               /(1 + Net.γ*normW2[i+1]))
    end
    z_free[1] = activation((w1x_plus_b1 + Net.γ*wT[2]*z_free[2])/(1 + Net.γ*normW2[2]))
    return z_clamped, z_free
end

function get_z_ultra_fast(w1x_plus_b1, normW2, wT, z_clamped, z_free, Net, activation)
    z_clamped[1] = activation((w1x_plus_b1 + Net.γ*wT[2]*z_clamped[end])/(1 + Net.γ*normW2[2]))
    @inbounds for i=2:Net.nLayers-2
        z_clamped[i] = activation((Net.w[i]*z_clamped[i-1] + Net.b[i] + Net.γ*wT[i+1]*z_clamped[end])
                                  /(1 + Net.γ*normW2[i+1]))
    end
    L = Net.nLayers - 1
    z_free[L] = Net.w[L]*z_clamped[L-1] + Net.b[L]
    @inbounds for i in (Net.nLayers-2):-1:2
        z_free[i] = activation((Net.w[i]*z_clamped[i-1] + Net.b[i] + Net.γ*wT[i+1]*z_free[end])
                               /(1 + Net.γ*normW2[i+1]))
    end
    z_free[1] = activation((w1x_plus_b1 + Net.γ*wT[2]*z_free[end])/(1 + Net.γ*normW2[2]))
    return z_clamped, z_free
end

function get_∇w(∇w, ∇b, x, z_clamped, z_free)
    # First layers weights
    ∇b[1] = z_free[1] - z_clamped[1]
    ∇w[1] = ∇b[1]*x'
 

    #subsequent layers
    for i=2:Net.nLayers-1
		    ∇b[i] = Net.γ^(i-1)*(z_free[i] - z_clamped[i])
		    ∇w[i] = Net.γ^(i-1) * ((z_free[i]*z_free[i-1]' - z_clamped[i]*z_clamped[i-1]')
                              + Net.w[i]*(z_clamped[i-1]'*z_clamped[i-1] - z_free[i-1]'*z_free[i-1]))

		    # Cheaper approximation
		    # ∇b[i] = γ^(i-1)*(z_free[i] - z_clamped[i])
		    # ∇w[i] = (∇b[i]*z_clamped[i-1]') + Net.γ^(i-1)*Net.w[i]*(z_clamped[i-1]'*(z_clamped[i-1] - z_free[i-1]))

        # Closed form solution
        # ∇w[i] = (z_free[i]*z_free[i-1]' - z_clamped[i]*z_clamped[i-1]')/(z_clamped[i-1]'*z_clamped[i-1] - z_free[i-1]'*z_free[i-1])
    end

    return ∇w, ∇b
end
function predict_direct_random_feedback(Net, fb, x, y, batchsize, numIter, activation)
	  """Train an MLP. Training is parallel across datapoints."""

	  L = Net.nLayers-1
	  nNeurons = Net.nNeurons

	  nBatches = trunc(Int64, length(y) / batchsize)
	  nSamples = length(y) - (length(y)%batchsize)
	  #Allocate array for activations
	  z = [[zeros(Float64, N) for N in nNeurons[2:end]] for i=1:batchsize]

	  # And a variable for counting batch number
	  batchIndex::Int64 = 0
	  correct::Int64 = 0
    normW2 = [norm(w)^2 for w in Net.w]

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
            # z[n] = forward(z[n], Net, activation, w1x_plus_b1)

			      z[n] = get_z_direct_random_feedback(w1x_plus_b1, normW2, fb, z[n], Net, activation, numIter, false)
            # z[n] = forward(z[n], Net, activation, w1x_plus_b1)
		    end

		    # Correct equals true if prediction is correct. Otherwise equals false.
		    z_out = [zk[end] for zk in z]
		    correct += sum((argmax.(z_out).-1).==yBatch)

	  end

	  Av_accuracy = 100*correct/nSamples

	  println("\nTraining finished\n")

	  return Av_accuracy
end
function predict_FF(Net, wT, x, y, batchsize, numIter, activation)
	  """Train an MLP. Training is parallel across datapoints."""

	  L = Net.nLayers-1
	  nNeurons = Net.nNeurons

	  nBatches = trunc(Int64, length(y) / batchsize)
	  nSamples = length(y) - (length(y)%batchsize)
	  #Allocate array for activations
	  z = [[zeros(Float64, N) for N in nNeurons[2:end]] for i=1:batchsize]

	  # And a variable for counting batch number
	  batchIndex::Int64 = 0
	  correct::Int64 = 0
    normW2 = [norm(w)^2 for w in Net.w]

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
            z[n] = forward(z[n], Net, activation, w1x_plus_b1)
			      # z[n] = get_z(w1x_plus_b1, normW2, wT, z[n], Net, activation, numIter, false)
		    end

		    # Correct equals true if prediction is correct. Otherwise equals false.
		    z_out = [zk[end] for zk in z]
		    correct += sum((argmax.(z_out).-1).==yBatch)

	  end

	  Av_accuracy = 100*correct/nSamples

	  println("\nTraining finished\n")

	  return Av_accuracy
end

function predict(Net, wT, x, y, batchsize, numIter, activation)
	  """Train an MLP. Training is parallel across datapoints."""

	  L = Net.nLayers-1
	  nNeurons = Net.nNeurons

	  nBatches = trunc(Int64, length(y) / batchsize)
	  nSamples = length(y) - (length(y)%batchsize)
	  #Allocate array for activations
	  z = [[zeros(Float64, N) for N in nNeurons[2:end]] for i=1:batchsize]

	  # And a variable for counting batch number
	  batchIndex::Int64 = 0
	  correct::Int64 = 0
    normW2 = [norm(w)^2 for w in Net.w]

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
            # z[n] = forward(z[n], Net, activation, w1x_plus_b1)
			      z[n] = get_z(w1x_plus_b1, normW2, wT, z[n], Net, activation, numIter, false)
		    end

		    # Correct equals true if prediction is correct. Otherwise equals false.
		    z_out = [zk[end] for zk in z]
		    correct += sum((argmax.(z_out).-1).==yBatch)

	  end

	  Av_accuracy = 100*correct/nSamples

	  println("\nTraining finished\n")

	  return Av_accuracy
end

function update_weights_ADAM(Net, ∇w, ∇b, M_w, M_b, V_w, V_b)
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

    Net.w -= Net.η * step_w
    Net.b -= Net.η * step_b

end

function trainThreads(Net, xTrain, yTrain, xTest, yTest, batchsize, testBatchsize, nEpochs, numIter, activation, random_feedback, direct_random_feedback, outpath)
	  """Train an MLP. Training is parallel across datapoints."""

	  L = Net.nLayers-1
	  nNeurons = Net.nNeurons

	  nBatches = trunc(Int64, length(yTrain) / batchsize)
	  nSamples = length(yTrain) - (length(yTrain)%batchsize)
	  #Allocate arrays to save training metrics to
	  E_free_batch = zeros(Float64, batchsize)
	  E_clamped_batch = zeros(Float64, batchsize)
	  z_free_batch = [[zeros(Float64, N) for N in nNeurons[2:end]] for i=1:batchsize]
	  z_clamped_batch = [[zeros(Float64, N) for N in nNeurons[2:end]] for i=1:batchsize]
	  ∇w_batch = [[zeros(Float64, (nNeurons[n+1], nNeurons[n])) for n = 1:L] for i=1:batchsize]
	  ∇b_batch = [[zeros(Float64, (nNeurons[n+1])) for n = 1:L] for i=1:batchsize]

	  # Arrays for ADAM
    M_w = [zeros(Float64, (Net.nNeurons[i+1], Net.nNeurons[i])) for i = 1:Net.nLayers-1]
	  M_b = [zeros(Float64, (Net.nNeurons[i+1])) for i = 1:Net.nLayers-1]
	  V_w = [zeros(Float64, (Net.nNeurons[i+1], Net.nNeurons[i])) for i = 1:Net.nLayers-1]
	  V_b = [zeros(Float64, (Net.nNeurons[i+1])) for i = 1:Net.nLayers-1]

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
		    E_free::Float64 = 0
		    E_clamped::Float64 = 0


		    # Loop through mini batches
		    for i = 1:batchsize:nSamples
			      batchIndex += 1
			      start = i
			      stop = start + batchsize - 1
			      #if stop>nSamples
			      #    break
			      #end
			      xBatch = @view xTrain[start:stop]
			      yBatch = @view yTrain[start:stop]

            if random_feedback
                fb = Net.g
            else
                fb = [w' for w in Net.w]
            end
            if direct_random_feedback
                fb = Net.u
            end

            # precompute
            normW2 = [norm(w)^2 for w in Net.w]

            Threads.@threads for n = 1:batchsize
                # Precompute  w1x and w1x+b1
				        w1x_plus_b1::Array{Float64,1} = Net.w[1]*xBatch[n] + Net.b[1]
                # Get activations
							  z_clamped_batch[n][end] = [yBatch[n]+1==k for k=1:10]
                # z_clamped_batch[n] = get_z(w1x_plus_b1, normW2, fb, z_clamped_batch[n], Net, activation, numIter, true);
                # z_free_batch[n] = get_z(w1x_plus_b1, normW2, fb, z_free_batch[n], Net, activation, numIter, false)
                # z_clamped_batch[n], z_free_batch[n] = get_z_ultra_fast(w1x_plus_b1, normW2, fb, z_clamped_batch[n], z_free_batch[n], Net, ReLU)
                z_clamped_batch[n], z_free_batch[n] = get_z_fast2(w1x_plus_b1, normW2, fb, z_clamped_batch[n], z_free_batch[n], Net, ReLU, numIter)


                # Get Gradient and update weights
                ∇w_batch[n], ∇b_batch[n] = get_∇w(∇w_batch[n], ∇b_batch[n], xBatch[n], z_clamped_batch[n], z_free_batch[n])

                # Get energies
				        E_free_batch[n] = energy(z_free_batch[n], xBatch[n], w1x_plus_b1, normW2, Net)
				        E_clamped_batch[n] = energy(z_clamped_batch[n], xBatch[n], w1x_plus_b1, normW2, Net)

			      end

			      ∇w = mean(∇w_batch)
			      ∇b = mean(∇b_batch)
			      E_free += sum(E_free_batch)
			      E_clamped += sum(E_clamped_batch)
			      # Correct equals true if prediction is correct. Otherwise equals false.
			      z_out = [zk[end] for zk in z_free_batch]
			      correct += sum((argmax.(z_out).-1).==yBatch)

			      # Net.w -= η * ∇w
			      # Net.b -= η * ∇b
			      update_weights_ADAM(Net, ∇w, ∇b, M_w, M_b, V_w, V_b)
            #Net.b = [min.(b, 0) for b in Net.b]# Clamp bias from above

			      print("\r$(@sprintf("%.2f", 100*batchIndex/nBatches))% complete")

		    end

		    acc_train = 100*correct/nSamples
		    Av_E_free = E_free/nSamples
		    Av_E_clamped = E_clamped/nSamples
		    Av_J = Av_E_clamped - Av_E_free
        av_z1_pos = mean([100*sum(z_free_batch[n][1].>0)/Net.nNeurons[2] for n=1:batchsize])
        sum_z1 = mean([sum(z_free_batch[n][1]) for n=1:batchsize])

        if random_feedback
            fb = Net.g
        else
            fb = [w' for w in Net.w]
        end
        if direct_random_feedback
            fb = Net.u
        end

		    acc_test = predict(Net, fb, xTest, yTest, testBatchsize, numIter, ReLU)
        # acc_test = predict_direct_random_feedback(Net, fb, xTest, yTest, testBatchsize, numIter, ReLU)

		    push!(Net.History["acc_train"], acc_train)
		    push!(Net.History["acc_test"], acc_test)
		    push!(Net.History["J"], Av_J)
		    push!(Net.History["E_free"], Av_E_free)
		    push!(Net.History["E_clamped"], Av_E_clamped)
		    push!(Net.History["av_z1_pos"], av_z1_pos)
        push!(Net.History["sum_z1"], sum_z1)


		    t1 = time()
		    # println("\nEpoch $epoch/$nEpochs: \tProcessing time: $(@sprintf("%.6f", t1-t0))\n")
		    println("\nav. metrics: \tacc_train: $(@sprintf("%.2f", acc_train))%\tacc_test: $(@sprintf("%.2f", acc_test))%\tJ: $(@sprintf("%.8f", Av_J)) \tz1>0: $(@sprintf("%.2f", av_z1_pos))%\tsum(z1): $(@sprintf("%.4f", sum_z1))\tProc. time: $(@sprintf("%.2f", t1-t0))")

		    push!(Net.History["runTimes"], t1-t0)
		    FileIO.save("$outpath", "Net", Net)
	  end
	  println("\nTraining finished")
end

