using Flux
using Flux.Losses: mse
using LinearAlgebra


struct rbf{S<:AbstractArray, T<:AbstractArray}
    v::S
    ϵ::T
end

function rbf(in::Integer, out::Integer;
             initv = Flux.glorot_uniform, initϵ = 1.0f0)
    v = [initv(in) for k=1:out]
    ϵ = [initϵ for k=1:out]
    return rbf(v, ϵ)
end

Flux.@functor rbf

function (a::rbf)(x::AbstractArray)
    v, ϵ = a.v, a.ϵ
    [exp(-ϵ[k]*norm(a.v[k] - x)) for k=1:length(v)]
end




NetRBF = Chain(
    Dense(10, 7, σ),
    rbf(7, 8),
    Dense(8, 5),
    softmax)

a = NetRBF(rand(10)) # => 5-element vector

struct proto{S<:AbstractArray, T<:AbstractArray}
    v::S
    r::T
end

function proto(in::Integer, out::Integer;
             initv = Flux.glorot_uniform, initr = 5.0f0)
    v = [initv(in) for k=1:out]
    r = [initr for k=1:out]
    return proto(v, r)
end

Flux.@functor proto

function (a::proto)(x::AbstractArray)
    v, r = a.v, a.r
    [(norm(a.v[k] - x)<r[k]) for k=1:length(v)]
end


NetProto = Chain(
    Dense(10, 7, σ),
    proto(7, 8),
    Dense(8, 5),
    softmax)

x = randn(10)
a = NetProto(x) # => 5-element vector

y = [0, 0, 0, 0, 1.0]
ps = Flux.params(NetProto)
∇θ = gradient(() -> mse(NetProto(x), y), ps) # compute gradient
