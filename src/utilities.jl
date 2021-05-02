@kwdef struct Args
    # Network specific parameters
    nNeurons::Array{Int32, 1}
    activation::Vector{Function}
    highLim::Array{Float32,1}
    nLayers::Int32

    # Hyper parameters
    batchsize::Int = 64
    test_batchsize::Int = 10000
    nOuterIterations::Int = 2
    nInnerIterations::Int = 5 # use ~10-15 for CIFAR10
    nEpochs::Int = 10

    # misc parameters
    outpath::String = "../networks/Net.jld2"
    numThreads::Int = Threads.nthreads()

end

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
	      "J_train" => Float32[],
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
If low=0 and high=1.0, then you have Hard Sigmoid."""
function Clamp(z, low=0.0, high=Inf)
    return min(high, max(low, z))
end

"""Hard sigmoid"""
HS(z) = Clamp(z, 0, 1.0)

"""Feed-forward pass"""
function forward!(z, Net, activation, w1x_plus_b1)
    z[1] = activation[1].(w1x_plus_b1)

    for i = 2:Net.nLayers-1
        z[i] =  activation[i].(Net.w[i] * z[i-1] .+ Net.b[i])
    end

    L = Net.nLayers
    z[L] = activation[L].(Net.w[L] * z[L-1] .+ Net.b[L])
end

