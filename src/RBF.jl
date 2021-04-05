using Flux
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




model = Chain(
    Dense(10, 7, σ),
    rbf(7, 8),
    Dense(8, 5),
    softmax)

a = model(rand(10)) # => 5-element vector

