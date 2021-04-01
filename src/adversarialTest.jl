# Loading Dataset	
xtrain, ytrain = MLDatasets.MNIST.traindata(Float32)
xtest, ytest = MLDatasets.MNIST.testdata(Float32)

# Reshape Data in order to flatten each image into a linear array
xtrain = Flux.flatten(xtrain)
xtest = Flux.flatten(xtest)

# One-hot-encode the labels
ytrain, ytest = onehotbatch(ytrain, 0:9), onehotbatch(ytest, 0:9)

# Tweaked version of Adversarial.jls PGD function (to make it work on CPU).
function PGD2(model, loss, x, y; ϵ = 10, step_size = 0.001, iters = 100, clamp_range = (0, 1))
    x_adv = clamp.(x + ((randn(Float32, size(x)...)) * Float32(step_size)), clamp_range...); # start from the random point
    δ = chebyshev(x, x_adv)
    iter = 1; while (δ < ϵ) && iter <= iters
        x_adv = FGSM(model, loss, x_adv, y; ϵ = step_size, clamp_range = clamp_range)
        δ = chebyshev(x, x_adv)
        iter += 1
    end
    return x_adv
end

k = 86
x_adv = PGD2(NetFlux, mse, xTest[k], yTest[k])


pred = NetFlux(xTest[k])
println(argmax(pred))

pred = NetFlux(x_adv)
println(argmax(pred))
