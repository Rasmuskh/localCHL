using Plots; # For plotting
pyplot(); # Choosing backend for Plots

function plot_filters(Net, nRows, nCols, width, height, imwidth, imheight)
    p = []

    W=[reshape(Net.w[1][i,:], (imwidth, imheight)) for i=1:Net.nNeurons[2]]
    if nCols*nRows>length(W)
        for i=length(W)+1:nCols*nRows
            push!(W, zeros(size(W[1])))
        end
    end
    for i=1:length(W)
        # pDummy = plot(heatmap(W[i], color = :grayscale), aspect_ratio=:equal, grid=false, axiscolor=nothing, size=(nCols,nRows))
        pDummy = plot(heatmap(rotr90(W[i], 3), color = :grays), border=:none, aspect_ratio=:equal, grid=false, axiscolor=nothing)
        push!(p, pDummy)
    end

    P = plot(p..., legend = false, layout = (nCols,nRows), xticks=false, yticks=false, colorbar=false)
    P = plot!(size=(width,height))
    display(P)
end

function plot_filters_struct(Net, nRows, nCols, width, height, imwidth, imheight)
    p = []

    W=[reshape(Net.w[1][i,:], (imwidth, imheight)) for i=1:Net.nNeurons[2]]
    if nCols*nRows>length(W)
        for i=length(W)+1:nCols*nRows
            push!(W, zeros(size(W[1])))
        end
    end
    for i=1:length(W)
        # pDummy = plot(heatmap(W[i], color = :grayscale), aspect_ratio=:equal, grid=false, axiscolor=nothing, size=(nCols,nRows))
        pDummy = plot(heatmap(rotr90(W[i], 3), color = :grays), border=:none, aspect_ratio=:equal, grid=false, axiscolor=nothing)
        push!(p, pDummy)
    end

    P = plot(p..., legend = false, layout = (nCols,nRows), xticks=false, yticks=false, colorbar=false)
    P = plot!(size=(width,height))
    display(P)
end

function plot_key(Net, key)
    x=Array(1:length(Net.History[key]))
    P=plot(x, Net.History[key], lw=2, ylabel="key", xlabel="epoch")
    display(P)
end
