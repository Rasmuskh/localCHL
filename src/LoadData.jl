using MLDatasets #For loading the data
using NPZ #For reading K-MNIST data from .npy (numpy) files
using Statistics # For normalizing MNIST data

function loadData(dataset, trainSamples, testSamples)
    if dataset=="CIFAR10_color"
        #Training data
        yTrain = CIFAR10.trainlabels()
        xTrain = CIFAR10.traintensor()
        xTrain = convert(Array{dType}, xTrain)
        xTrain = [vec(vcat(xTrain[:, :, 1, i], xTrain[:, :, 2, i], xTrain[:, :, 3, i])) for i=1:length(yTrain)]

        #Testing data
        yTest = CIFAR10.testlabels()
        xTest = CIFAR10.testtensor()
        xTest = convert(Array{dType}, xTest)
        xTest = [vec(vcat(xTest[:, :, 1, i], xTest[:, :, 2, i], xTest[:, :, 3, i])) for i=1:length(yTest)]

    elseif dataset=="CIFAR10_gray"
        #Training data
        yTrain = CIFAR10.trainlabels()
        xTrain = CIFAR10.traintensor()
        xTrain = convert(Array{dType}, xTrain)
        xTrain = [vec(xTrain[:, :, 1, i] + xTrain[:, :, 2, i] + xTrain[:, :, 3, i])/3 for i=1:length(yTrain)]

        #Testing data
        yTest = CIFAR10.testlabels()
        xTest = CIFAR10.testtensor()
        xTest = convert(Array{dType}, xTest)
        xTest = [vec(xTest[:, :, 1, i] + xTest[:, :, 2, i] + xTest[:, :, 3, i])/3 for i=1:length(yTest)]


    elseif dataset=="MNIST"
        #Training data
        yTrain = MNIST.trainlabels()
        xTrain = MNIST.traintensor()
    	  xTrain = convert(Array{dType}, xTrain)
        xTrain = [vec(xTrain[:, :, i]) for i=1:length(yTrain)]
        # xTrain = [vec((xTrain[:,:,i].-0.1307)./0.3081) for i=1:length(yTrain)];

        #Testing data
        yTest = MNIST.testlabels()
    	  xTest = MNIST.testtensor()
    	  xTest = convert(Array{dType}, xTest)
        xTest = [vec(xTest[:, :, i]) for i=1:length(yTest)]
        # xTest = [vec((xTest[:,:,i].-0.1307)./0.3081) for i=1:length(yTest)];

    elseif dataset=="FMNIST"
        #Training data
        yTrain = FashionMNIST.trainlabels()
        xTrain = FashionMNIST.traintensor()
        xTrain = convert(Array{dType}, xTrain)
        xTrain = [vec(xTrain[:, :, i]) for i=1:length(yTrain)]
        #Testing data
        yTest = FashionMNIST.testlabels()
        xTest = FashionMNIST.testtensor()
        xTest = convert(Array{dType}, xTest)
        xTest = [vec(xTest[:, :, i]) for i=1:length(yTest)]

    elseif dataset=="KMNIST"
        #Training data
        yTrain = npzread("/home/rasmus/Documents/KMNIST/kmnist-train-labels/arr_0.npy")
        xTrain = npzread("/home/rasmus/Documents/KMNIST/kmnist-train-imgs/arr_0.npy")
        xTrain = convert(Array{dType}, xTrain)
        # xTrainMean = mean(xTrain)
        # xTrainStd = std(xTrain)
        xTrain = [vec(xTrain[i, :, :]) for i=1:length(yTrain)]
        xTrain = xTrain/255
        # xTrain = [(xTrain[i].-xTrainMean)./xTrainStd for i=1:length(yTrain)]
        #Testing data
        yTest = npzread("/home/rasmus/Documents/KMNIST/kmnist-test-labels/arr_0.npy")
        xTest = npzread("/home/rasmus/Documents/KMNIST/kmnist-test-imgs/arr_0.npy")
        xTest = convert(Array{dType}, xTest)
        xTest = [vec(xTest[i, :, :]) for i=1:length(yTest)]
        xTest = xTest/255
        # xTestMean = mean(xTest)
        # xTestStd = std(xTest)
        # xTest = [(xTest[i].-xTestMean)./xTestStd for i=1:length(yTest)]
    else
        throw(ArgumentError("Invalid dataset name. Valid options are: CIFAR10_color, CIFAR10_gray, MNIST, KMNIST, FMNIST."))
    end
    xTrain = xTrain[1:trainSamples]
    yTrain = yTrain[1:trainSamples]
    xTest = xTest[1:testSamples]
    yTest = yTest[1:testSamples]
    return (xTrain, yTrain, xTest, yTest)
end
