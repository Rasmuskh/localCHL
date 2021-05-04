"""Struct for holding various arguments for the train function"""
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

"""Network struct"""
mutable struct Net_CHL
	  w::Array{Array{Float32, 2},1}
	  b::Array{Array{Float32, 1},1}
    highLim::Array{Float32,1}
	  History::Dict{String, Array{Float32,1}}
end

"""Initialize network weights and allocate arrays for training metric history."""
function init_network(nNeurons, highLim, init_mode="glorot_normal")
	  nLayers = length(nNeurons) - 1
    if init_mode == "glorot_normal"
        w = [1/sqrt(nNeurons[i] + nNeurons[i+1]) * randn(Float32, (nNeurons[i+1], nNeurons[i])) for i = 1:nLayers]
    elseif init_mode == "glorot_uniform"
        w = [(rand(Float32, (nNeurons[i+1], nNeurons[i])) .- 0.5) * sqrt(6.0/(nNeurons[i] + nNeurons[i+1])) for i = 1:nLayers]
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
	  Net = Net_CHL(w, b, highLim, History)
	  println("Network initialized")
	  return Net
end

"""Convert an LPOM/local-CHL model to a Flux Chain model.
TODO (perhaps): refactor code to use Flux Chain model for LPOM."""
function LPOM_to_Flux(Net, activation, nLayers)
    D = []
    for i in 1:nLayers
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
function forward!(z, Net, activation, w1x_plus_b1, nLayers)
    z[1] = activation[1].(w1x_plus_b1)
    for i = 2:nLayers
        z[i] =  activation[i].(Net.w[i] * z[i-1] .+ Net.b[i])
    end
end

"""Fast dot product utilizing avx. Source LoopVectorization.jl:
https://juliasimd.github.io/LoopVectorization.jl/latest/examples/dot_product/"""
function dotavx(a, b)
    s = zero(eltype(a))
    @avx for i ∈ eachindex(a, b)
        s += a[i] * b[i]
    end
    s
end

"""Fast squared sum of vector elements (self-dot product) utilizing avx. Source LoopVectorization.jl:
https://juliasimd.github.io/LoopVectorization.jl/latest/examples/dot_product/"""
function selfdotavx(a)
    s = zero(eltype(a))
    @avx for i ∈ eachindex(a)
        s += a[i] * a[i]
    end
    s
end
