using Pkg; for p in ["Knet"]; haskey(Pkg.installed(),p) || Pkg.add(p); end #Knet installation to use the MNIST dataset
using Knet, Printf, Random
import Knet: minibatch, accuracy
include(Knet.dir("data", "mnist.jl"))

function minibatch(X, Y, bs=100)
    data = Any[]
    for i=1:bs:size(X,2)
        j = min(i+bs-1,size(X,2))
        push!(data, ((X[:,i:j]), Y[:,i:j]))
    end
    return data
end

function init_params(ninputs, noutputs)
    b = fill(0.0,(noutputs,1))
    W = fill(0.0,(noutputs,ninputs))
    W = randn((noutputs,ninputs))*0.001
    return W, b
end

function softmax_forw(W, b, x)
    y = (W * x) .+ b
    probs = exp.(y) ./ sum(exp.(y), dims =1)
    return probs
end

function softmax_back_and_loss(W, b, x, ygold)
    prob_softmax = softmax_forw(W, b, x)
    loss  = -sum(ygold .* log.(prob_softmax)) ./ size(prob_softmax,2)        
    gw = (W .* loss) ./ size(W,2)
    gb = sum(prob_softmax-ygold,dims = 2) /size(prob_softmax,2)
    return loss, gw, gb
end

function grad_check(W, b, x, ygold)
    # numeric_gradient()
    # calculates and returns numeric gradients of model weights (gw,gb)
    function numeric_gradient()
        epsilon = 0.0001

        gw = zeros(size(W)) # gradient of W
        gb = zeros(size(b)) # gradient of b

        gw = ((W .* (gw .+ epsilon)) - (W  .* (gw .- epsilon))) / (2 * epsilon)
        gb = ((b .* (gb .+ epsilon)) - (b  .* (gb .- epsilon))) / (2 * epsilon)  
        
        return gw, gb
    end

    _,gradW,gradB = softmax_back_and_loss(W, b, x, ygold)
    gw, gb = numeric_gradient()

    diff = sqrt(sum((gradW - gw) .^ 2) + sum((gradB - gb) .^ 2))
    println("Diff: $diff")
    if diff < 1e-7
        println("Gradient Checking Passed")
    else
        println("Diff must be < 1e-7")
    end
end

function train(W, b, data, lr=0.15)
    totalcost = 0.0
    numins = 0
    for (x, y) in data
        cost,grad_W, grad_b =softmax_back_and_loss(W, b, x, y)
        W  .= W - lr * grad_W
        b  .= b - lr * grad_b
        totalcost = totalcost + cost .* size(x,2)
        numins = numins .+ size(x,2)
    end
    avgcost = totalcost / numins
end

function accuracy(ygold, ypred)
    correct = 0.0
    for i=1:size(ygold, 2)
        correct += findmax(ygold[:,i]; dims=1)[2] == findmax(ypred[:, i]; dims=1)[2] ? 1.0 : 0.0
    end
    return correct / size(ygold, 2)
end

function main()
    Random.seed!(12345)
    ninputs = 28 * 28
    noutputs = 10

    xtrn, ytrn, xtst, ytst = mnist() # loading the data
    xtrn = reshape(xtrn, 784, 60000)
    xtst = reshape(xtst, 784, 10000)

    function to_onehot(x)
        onehot = zeros(10, 1)
        onehot[x, 1] = 1.0
        return onehot
    end

    ytrn = hcat(map(to_onehot, ytrn)...)
    ytst = hcat(map(to_onehot, ytst)...)

    # STEP 1: Create minibatches
    #   Complete the minibatch function
    #   It takes the input matrix (X) and gold labels (Y)
    #   returns list of tuples contain minibatched input and labels (x, y)
    bs = 100
    trn_data = minibatch(xtrn, ytrn, bs)

    # STEP 2: Initialize parameters
    #   Complete init_params function
    #   It takes number of inputs and number of outputs(number of classes)
    #   It returns randomly generated W matrix and bias vector
    #   Sample from N(0, 0.001)

    W, b = init_params(ninputs, noutputs)

    # STEP 3: Implement softmax_forw and softmax_back_and_loss
    #   softmax_forw function takes W, b, and data
    #   calculates predicted probabilities
    #
    #   softmax_back_and_loss function obtains probabilites by calling
    #   softmax_forw then calculates soft loss and gradients of W and b

    # STEP 4: Gradient checking
    #   Skip this part for the lab session.
    #   As with any learning algorithm, you should always check that your
    #   gradients are correct before learning the parameters.

    debug = true # Turn this parameter off, after gradient checking passed
    if debug
        grad_check(W, b, xtrn[:, 1:100], ytrn[:, 1:100])
    end

    lr = 0.15

    # STEP 5: Training
    #   The train function takes model parameters and the data
    #   Trains the model over minibatches
    #   For each minibatch, first cost and gradients are calculated then model parameters are updated
    #   train function returns the average cost per instance

    for i=1:50
        cost = train(W, b, trn_data, lr)
        pred = softmax_forw(W, b, xtrn)
        trnacc = accuracy(ytrn, pred)
        pred = softmax_forw(W, b, xtst)
        tstacc = accuracy(ytst, pred)
        @printf("epoch: %d softloss: %g trn accuracy: %g tst accuracy: %g\n", i, cost, trnacc, tstacc)
    end
end

main()