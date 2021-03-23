using JLD2
using FileIO
include("PlottingFunctions.jl")

Net = load("networks/MNIST-300-300-10.jld2")["Net"]
nRows = 30
nCols = 10

# @time plot_filters(Net, nRows, nCols, 1800, 600, 28, 28)

key = "acc_train"
x=Array(1:length(Net.History[key]))
P=plot(x, Net.History[key], lw=2, ylabel="key", xlabel="epoch")

key = "acc_test"
P=plot!(x, Net.History[key], lw=2, ylabel="key", xlabel="epoch")
