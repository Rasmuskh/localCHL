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
	g = [sqrt(2 / nNeurons[i]) * randn(rng, Float64, (nNeurons[i+1], nNeurons[i])) for i = 1:nLayers-1]
	b = [zeros(Float64, (nNeurons[i+1])) for i = 1:nLayers-1]

	num_updates = 0
    # Network history dictionary
	History = Dict(
	"runTimes" => Float64[],
	"acc_train" => Float64[],
	"acc_test" => Float64[],
	"test_accuracy" => Float64[],
	"J" => Float64[],
	"E_free" => Float64[],
	"E_clamped" => Float64[],
    )
    
	# Network struct
	Net = Net_CHL(nNeurons, η, γ, nLayers, w, g, b, num_updates, History)
	println("Network initialized")
	return Net
end

function ReLU(PreActivation)
    """ReLU function: Clamps input between 0 and infinity"""
    return max.(0, PreActivation)
end

function energy(z, w0x_plus_b0, Net)
    a = w0x_plus_b0 - z[1]
    E = 0.5 * (a'*a - w0x_plus_b0'*w0x_plus_b0)

    for i = 2:Net.nLayers-1
        b = Net.w[i]*z[i-1] + Net.b[i]
        a = b - z[i]
        E += 0.5*Net.γ^(i-1)*(a'*a - b'*b)

    end
    return E
end

function forward(z, Net, activation, w1x_plus_b1)
    """Simple forward pass"""
    # z = [zeros(Float64, N) for N in Net["nNeurons"][2:end]]
    z[1] = activation(w1x_plus_b1)
    for i = 2:Net.nLayers-2
        z[i] =  activation(Net.w[i] * z[i-1] + Net.b[i])
    end
    #The final layer is linear
    L = Net.nLayers-1
    z[L] = Net.w[L] * z[L-1] + Net.b[L]
    return z
end


function get_z(w1x_plus_b1, wT, z, Net, activation, nIter, clamped)
    for u=1:nIter
        z[1] = w1x_plus_b1 + Net.γ*wT[2]*z[2]
        for i=2:Net.nLayers-2
            z[i] = Net.w[i]*z[i-1] + Net.b[i] + Net.γ*wT[i+1]*z[i+1]
            z[i] = activation(z[i])
        end
        if clamped==false
            L = Net.nLayers - 1
            z[L] = Net.w[L]*z[L-1] + Net.b[L]
            z[L] = activation(z[L])
        end

    end
    return z
end

function get_∇w(∇w, ∇b, x, z_clamped, z_free)
    ∇w[1] = z_free[1]*x' - z_clamped[1]*x'
    ∇b[1] = z_free[1] - z_clamped[1]

    for i=2:Net.nLayers-1
		∇b[i] = γ^(i-1)*(z_free[i] - z_clamped[i])
		∇w[i] = γ^(i-1)*(z_free[i]*z_free[i-1]' - z_clamped[i]*z_clamped[i-1]')
		
		# Cheaper approximation
		# ∇b[i] = γ^(i-1)*(z_free[i] - z_clamped[i])
		# ∇w[i] = ∇b[i]*z_clamped[i-1]'

    end
    
    return ∇w, ∇b
end

function predict(Net, x, y, batchsize, numIter, activation, random_feedback)
	"""Train an MLP. Training is parallel across datapoints."""

	L = Net.nLayers-1
	nNeurons = Net.nNeurons

	nBatches = trunc(Int64, length(y) / batchsize)
	nSamples = length(y) - (length(y)%batchsize)
	#Allocate array for activations
	z = [[zeros(Float64, N) for N in nNeurons[2:end]] for i=1:batchsize]

	#Shuffle data
	order = randperm(length(y))
	x = x[order]
	y = y[order]

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
        wT = [w' for w in Net.w]
		gT = [g' for g in Net.g]

        @inbounds @Threads.threads for n = 1:batchsize
        # Precompute  w1x and w1x+b1
			w1x_plus_b1::Array{Float64,1} = Net.w[1]*xBatch[n] + Net.b[1]
			# Get activations
			# if random_feedback == false
			# 	z[n] = get_z(w1x_plus_b1, wT, z[n], Net, activation, numIter, false)
			# else
				z[n] = get_z(w1x_plus_b1, gT, z[n], Net, activation, numIter, false)
			# end
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

function trainThreads(Net, xTrain, yTrain, xTest, yTest, batchsize, nEpochs, numIter, activation, outpath)
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
	# M_w = [zeros(Float64, (Net.nNeurons[i+1], Net.nNeurons[i])) for i = 1:Net.nLayers-1]
	# M_b = [zeros(Float64, (Net.nNeurons[i+1])) for i = 1:Net.nLayers-1]
	# V_w = [zeros(Float64, (Net.nNeurons[i+1], Net.nNeurons[i])) for i = 1:Net.nLayers-1]
	# V_b = [zeros(Float64, (Net.nNeurons[i+1])) for i = 1:Net.nLayers-1]

	#Loop across batches
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
			xBatch = xTrain[start:stop]
			yBatch = yTrain[start:stop]
			# wT = [w' for w in Net.w]
			gT = [g' for g in Net.g]

            @inbounds @Threads.threads for n = 1:batchsize
                
            # Precompute  w1x and w1x+b1
				w1x_plus_b1::Array{Float64,1} = Net.w[1]*xBatch[n] + Net.b[1]
                # Get activations
                # z_free_batch[n] = forward(z_free_batch[n], Net, activation, w1x_plus_b1)
				z_free_batch[n] = get_z(w1x_plus_b1, gT, z_free_batch[n], Net, activation, numIter, false)
				# z_free_batch[n] = get_z(w1x_plus_b1, wT, z_free_batch[n], Net, activation, numIter, false)
				# z_clamped_batch = deepcopy(z_free_batch)
				z_clamped_batch[n][end] = zeros(nNeurons[end]); z_clamped_batch[n][end][yBatch[n]+1] = 1
                z_clamped_batch[n] = get_z(w1x_plus_b1, gT, z_clamped_batch[n], Net, activation, numIter, true);
                # z_clamped_batch[n] = get_z(w1x_plus_b1, wT, z_clamped_batch[n], Net, activation, numIter, true);


                # Get Gradient and update weights
                ∇w_batch[n], ∇b_batch[n] = get_∇w(∇w_batch[n], ∇b_batch[n], xBatch[n], z_clamped_batch[n], z_free_batch[n])

                # Get energies
				E_free_batch[n] = energy(z_free_batch[n], w1x_plus_b1, Net)
				E_clamped_batch[n] = energy(z_clamped_batch[n], w1x_plus_b1, Net)

			end

			∇w = mean(∇w_batch)
			∇b = mean(∇b_batch)
			E_free += sum(E_free_batch)
			E_clamped += sum(E_clamped_batch)
			# Correct equals true if prediction is correct. Otherwise equals false.
			z_out = [zk[end] for zk in z_free_batch]
			correct += sum((argmax.(z_out).-1).==yBatch)

			Net.w -= η * ∇w
			Net.b -= η * ∇b
			# update_weights_ADAM(Net, ∇w, ∇b, M_w, M_b, V_w, V_b)

			print("\r$(@sprintf("%.2f", 100*batchIndex/nBatches))% complete")

		end

		acc_train = 100*correct/nSamples
		Av_E_free = E_free/nSamples
		Av_E_clamped = E_clamped/nSamples
		Av_J = Av_E_clamped - Av_E_free

		random_feedback = true
		acc_test = predict(Net, xTest, yTest, batchsize, numIter, ReLU, random_feedback)

		push!(Net.History["acc_train"], acc_train)
		push!(Net.History["acc_test"], acc_test)
		push!(Net.History["J"], Av_J)
		push!(Net.History["E_free"], Av_E_free)
		push!(Net.History["E_clamped"], Av_E_clamped)

		t1 = time()
		# println("\nEpoch $epoch/$nEpochs: \tProcessing time: $(@sprintf("%.6f", t1-t0))\n")
		println("\nav. metrics: \tacc_train: $(@sprintf("%.2f", acc_train))\tacc_test: $(@sprintf("%.2f", acc_test)) \tJ: $(@sprintf("%.8f", Av_J)) \tProc. time: $(@sprintf("%.2f", t1-t0))")

		push!(Net.History["runTimes"], t1-t0)
		FileIO.save("$outpath/Network_epoch_$epoch.jld2", "Net", Net)
	end
	println("\nTraining finished")
end

