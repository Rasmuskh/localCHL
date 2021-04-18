using Flux
using Flux.Losses: logitcrossentropy
using Flux: onehotbatch
using BenchmarkTools
using Tullio
using LinearAlgebra
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

Flux.@functor rbf
function (a::rbf)(x::AbstractArray)

    batchsize = size(x)[2]
    numIn = size(x)[1] 
    numOut = size(a.V)[2] # number of units in the RBF layer

    #= a.V and x are matrices, with the same number of rows, but different numbers of columns.
    Each column of x represents a different datapoint, and each column of V is a template/centroid.
    For each datapoint we compute the squared Euclidean distance to each of the columns of V. =#
    #d = a.V.-reshape(x, (numIn, 1, batchsize))
    #d = (sum(abs2, d, dims=1))
    V, β = a.V, a.β
    @tullio d[num_out, batch_size] := abs2(V[num_in, num_out] - x[num_in, batch_size])
    # Here size(d) = (1, numOut, batchsize) so next the singleton dimension is dropped
    # d = reshape(d, (numOut, batchsize))
    return exp.(-a.β.*d)
end

LinearAlgebra.BLAS.set_num_threads(8)
#nT = ccall((:openblas_get_num_threads64_, Base.libblas_name), Cint, ())
#println(nT)

# Generate a batch of 128 dummy datapoints
x = rand(Float32, 784,128)
y = onehotbatch(rand(0:9, 128), 0:9)

#Initialize network
rbfNet = Chain(rbf(784, 100), Dense(100, 10))
mlpNet = Chain(Dense(784, 100), Dense(100, 10))

println("Timing of forward pass in RBF network")
@btime rbfNet(x)
println("Timing of forward pass in MLP network")
@btime mlpNet(x)

θrbf = Flux.params(rbfNet)
println("Timing of backward pass in RBF network")
@btime ∇θrbf = gradient(() -> logitcrossentropy(rbfNet(x), y), θrbf) # compute gradient

θmlp = Flux.params(mlpNet)
println("Timing of backward pass in MLP network")
@btime ∇θmlp = gradient(() -> logitcrossentropy(mlpNet(x), y), θmlp) # compute gradient

