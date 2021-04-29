mutable struct Net_CHL
    nNeurons::Array{Int32, 1}
	  nLayers::Int32
	  w::Array{Array{Float32, 2},1}
	  b::Array{Array{Float32, 1},1}
    highLim::Array{Float32,1}
	  num_updates::Int32
	  History::Dict{String, Array{Float32,1}}
end


function init_network(nNeurons, highLim, init_mode="glorot_normal")
	  """Initialize network weights and allocate arrays for training metric history."""
	  nLayers = length(nNeurons) - 1
    if init_mode == "glorot_normal"
        w = [1/sqrt(nNeurons[i] + nNeurons[i+1]) * randn(rng, Float32, (nNeurons[i+1], nNeurons[i])) for i = 1:nLayers]
    elseif init_mode == "glorot_uniform"
        w = [(rand(rng, Float32, (nNeurons[i+1], nNeurons[i])) .- 0.5) * sqrt(6.0/(nNeurons[i] + nNeurons[i+1])) for i = 1:nLayers]
    end

	  b = [zeros(Float32, (nNeurons[i+1])) for i = 1:nLayers]

	  num_updates = 0
    # Network history dictionary
	  History = Dict(
	      "runTimes" => Float32[],
	      "acc_train" => Float32[],
	      "acc_test" => Float32[],
	      "J" => Float32[],
    )

	  # Network struct
	  Net = Net_CHL(nNeurons, nLayers, w, b, highLim, num_updates, History)
	  println("Network initialized")
	  return Net
end

function LPOM_to_Flux(Net, activation)
    D = []
    for i in 1:Net.nLayers
        w = Net.w[i]
        b = Net.b[i]
        Dummy = Dense(w, b, activation[i])#, initb=Net.b[i])
        push!(D, Dummy)
    end
    fluxNet = Chain(D...)

    return fluxNet
end

"""Clamp inputs. if low=0 and high=Inf then you have ReLU.
If low=0 and high=Inf, then you have Hard Sigmoid."""
function Clamp(z, low=0.0, high=Inf)
    return min(high, max(low, z))
end

"""Feed-forward pass"""
function forward!(z, Net, activation, w1x_plus_b1)
    z[1] = activation[1].(w1x_plus_b1)

    for i = 2:Net.nLayers-1
        z[i] =  activation[i].(Net.w[i] * z[i-1] .+ Net.b[i])
    end

    L = Net.nLayers
    z[L] = activation[L].(Net.w[L] * z[L-1] .+ Net.b[L])
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

