using Flux
using Flux.Losses: mse
using Flux: onehotbatch, onecold, @epochs
using Flux.Data: DataLoader
using Tullio
using LinearAlgebra
using Random; Random.seed!(3323); rng = MersenneTwister(12333)
using MLDatasets
using BenchmarkTools

function getdata(batchsize, trainsamples, testsamples)
    # Loading Dataset	
    xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
    xtrain = xtrain[:,:,1:trainsamples]; ytrain = ytrain[1:trainsamples]
    xtest, ytest = MLDatasets.MNIST.testdata(Float32)
    xtest = xtest[:,:,1:testsamples]; ytest = ytest[1:testsamples]

    # Reshape Data in order to flatten each image into a linear array
    xtrain = Flux.flatten(xtrain)
    xtest = Flux.flatten(xtest)

    # One-hot-encode the labels
    ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

    # Create DataLoaders (mini-batch iterators)
    train_loader = DataLoader((xtrain, ytrain), batchsize=batchsize, shuffle=true)
    test_loader = DataLoader((xtest, ytest), batchsize=batchsize)

    return train_loader, test_loader
end

"""
Initialize weights as samples from the training set (useful as a starting point for kmeans clustering).
#TODO: rewrite and optimize this function.
"""
function init_template!(ps)
    xTrain, yTrain, xTest, yTest = loadData("MNIST", 60000, 0)
    nProto = length(ps[2])
    nOut = length(unique(yTrain))
    n = Int64(round(nProto/nOut-0.5))
	  for i=1:nOut
		    tmp=xTrain[yTrain.==(i-1)][1:n]
        #Net.v[i*n - (n-1):i*n] = tmp
        ps[1][:,i*n - (n-1):i*n] = hcat(tmp...)
	  end
end


"""
Kmeans initialization of first layer weights.
#TODO: rewrite and optimize this function.
"""
function k_means!(ps, numIter)
    xTrain, yTrain, xTest, yTest = loadData("MNIST", 60000, 0)
    init_template!(ps)
    dist = [zeros(Float32, length(ps[2])) for k=1:length(xTrain)]
    labels = zeros(Int16, length(xTrain))

    # Loop for u iterations of k-means clustering
    for u=1:numIter
        print("\rK-means initialization: $(100*u/numIter)% complete.")
        # Loop through datapoints
        @Threads.threads for x_index=1:length(xTrain)
            j=1
            # loop through templates v and save distances between x and templates
            for  v in eachcol(ps[1])
                dist[x_index][j] = norm(xTrain[x_index]-v)
                j += 1
            end
            labels[x_index] = argmin(dist[x_index])
        end
        # Update centroids
        @Threads.threads for j=1:length(ps[2])
            ps[1][:,j] = mean(xTrain[labels.==j])
        end
    end
end

# ---Prototyping layer---
"""
Prototype layer.
"""
struct proto{F, S<:AbstractArray, T<:AbstractArray}
    V::S
    r::T
    α::Float32
    γ::Float32
    σ::F
end

# Hard sigmoid clamping between 0 and 1.
HS(t) = min(1,max(0, t))

"""
Initialization function
"""
function proto(in::Integer, out::Integer, σ=HS;
             initV = Flux.glorot_uniform, initr = 1.0f0, α = 1.0f0, γ=0.5f0)
    V = initV(in, out)
    r = initr*ones(Float32, out)
    return proto(V, r, α, γ, σ)
end

Flux.@functor proto
"""Make the layer work as a function as well as a struct"""
function (a::proto)(x::AbstractArray)
    V, r, α, γ, σ = a.V, a.r, a.α, a.γ, a.σ

    batchsize = size(x)[2]
    numIn = size(x)[1]
    numOut = size(a.V)[2]
    @tullio d[num_out, batch_size] := abs2(V[num_in, num_out] - x[num_in, batch_size])
    z = σ.((γ^(-1) .+ α*(1-γ)*r.*(r .- sqrt.(d)))./(1 .+ (α*(1-γ)^2*r.^2)))

    # TODO: remove these two lines. 
    # d = reshape(sqrt.(sum(abs2, a.V.-reshape(x, (numIn, 1, batchsize)), dims=1)), (numOut, batchsize))
    # z = σ.((γ^(-1) .+ α*(1-γ)*r.*(r .- d))./(1 .+ (α*(1-γ)^2*r.^2)))
    return z
end


#---Simplified prototyping layer---
struct protoSimple{F1, F2, S<:AbstractArray, T<:AbstractArray}
    V::S
    r::T
    α::Float32
    σ::F1
    E::F2
end

function EnergyProtoSimple(d, z1j, rj, α)
    return 0.5*max(0, d - rj + rj*α*z1j)^2 + 0.5*(z1j-1)^2
end

function protoSimple(in::Integer, out::Integer, σ=HS;
               initV = Flux.glorot_uniform, initr = 8.0f0, α = 0.5f0)
    V = initV(in, out)
    r = initr*ones(Float32, out)
    return protoSimple(V, r, α, σ, EnergyProtoSimple)
end

Flux.@functor protoSimple
function (a::protoSimple)(x::AbstractArray)
    V, r, α, σ = a.V, a.r, a.α, a.σ

    batchsize = size(x)[2]
    numIn = size(x)[1]
    numOut = size(a.V)[2]
    #d = reshape(sqrt.(sum(abs2, a.V.-reshape(x, (numIn, 1, batchsize)), dims=1)), (numOut, batchsize))
    #z = (1 .- α*r.*(d .- r))./(1 .+ (α^2*r.^2))
    @tullio d[num_out, batch_size] := abs2(V[num_in, num_out] - x[num_in, batch_size])
    z = σ.(1 .- α*r.*(sqrt.(d) .- r))./(1 .+ (α^2*r.^2))
    return z
end

Flux.@functor protoSimple
function (a::protoSimple)(x::AbstractArray, activation::Bool)
    V, r, α, σ = a.V, a.r, a.α, a.σ

    batchsize = size(x)[2]
    numIn = size(x)[1]
    numOut = size(a.V)[2]
    #d = reshape(sqrt.(sum(abs2, a.V.-reshape(x, (numIn, 1, batchsize)), dims=1)), (numOut, batchsize))
    #z = (1 .- α*r.*(d .- r))./(1 .+ (α^2*r.^2))
    @tullio d[num_out, batch_size] := abs2(V[num_in, num_out] - x[num_in, batch_size])
    z = (1 .- α*r.*(sqrt.(d) .- r))./(1 .+ (α^2*r.^2))
    return z
end

# ---Radial basis function layer---
"""Radial basis function layer"""
struct rbf{S<:AbstractArray, T<:AbstractArray}
    V::S
    β::T
end

function rbf(in::Integer, out::Integer;
             initV = Flux.glorot_uniform, β0 = 1.0f0)
    V = initV(in, out)
    β = β0*ones(Float32, out)
    return rbf(V, β)
end

# Flux.@functor rbf
# function (a::rbf)(x::AbstractArray)

#     batchsize = size(x)[2]
#     numIn = size(x)[1] 
#     numOut = size(a.V)[2] # number of units in the RBF layer

#     #= a.V and x are matrices, with the same number of rows, but different numbers of columns.
#     Each column of x represents a different datapoint, and each column of V is a template/centroid.
#     I For each datapoint I want the squared Euclidean distance to each of the columns of V. =#
#     d = a.V.-reshape(x, (numIn, 1, batchsize))
#     d = (sum(abs2, d, dims=1))
#     # Here size(d) = (1, numOut, batchsize) so next the singleton dimension is dropped
#     d = reshape(d, (numOut, batchsize))
#     return exp.(-a.β.*d)
# end

Flux.@functor rbf
function (a::rbf)(x::AbstractArray)

    batchsize = size(x)[2]
    numIn = size(x)[1] 
    numOut = size(a.V)[2] # number of units in the RBF layer
    V, β = a.V, a.β
    @tullio d[num_out, batch_size] := abs2(V[num_in, num_out] - x[num_in, batch_size])
    return exp.(-a.β.*d)
end

# ---Step radial basis function layer---
"""Radial basis function layer using a step like function instead of a gaussian"""
struct rbfStep{S<:AbstractArray, T<:AbstractArray}
    V::S
    β::T
    μ::T
end

# TODO: remove this in future version. It is just σ(-x).
# Zigmoid(x) = 1 / (1 + exp(x))

function rbfStep(in::Integer, out::Integer;
                 initV = Flux.glorot_uniform, β0=1.0f0, μ0=1.0f0)
    V = initV(in, out)
    β = β0*ones(Float32, out)
    μ = μ0*ones(Float32, out)
    return rbfStep(V, β, μ)
end

Flux.@functor rbfStep
function (a::rbfStep)(x::AbstractArray)

    batchsize = size(x)[2]
    numIn = size(x)[1]
    numOut = size(a.V)[2]

    # TODO: Old approach. Remove in the future.
    # d = reshape(sqrt.(sum(abs2, a.V.-reshape(x, (numIn, 1, batchsize)), dims=1)), (numOut, batchsize))
    # z = Zigmoid.(a.β.*(d.-a.μ))

    V, β, μ = a.V, a.β, a.μ
    @tullio d[num_out, batch_size] := abs2(V[num_in, num_out] - x[num_in, batch_size])
    z = σ.(-β.*(sqrt.(d).-μ))
    return z
end
