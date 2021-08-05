# -*- coding: utf-8 -*-
# -----------------------------------------------------------
# CS229: Machine Learning Assignment 3, Deep Learning & Unsupervised Learning
#
# author: Armand Sumo
# 
# email: armandsumo@gmail.com
# -----------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
from mnist import MNIST

def loadData():
    """

    trainData, trainLabels, testData, testLabels
    -------
    numpy ndarray
        Loads data from MNIST dataset. Requires MNIST library. 
        https://github.com/sorki/python-mnist

    """
    mndata = MNIST('data/')
    mndata.gz = True
    trainData, trainLabels = mndata.load_training()
    testData, testLabels = mndata.load_testing()
    return np.array(trainData), np.array(trainLabels), np.array(testData), np.array(testLabels)

def output(*args):
    sep = f"\n{'-'*37}"
    for x in args:
        print(x, sep, end="\n")   
        
def softmax(x):
    """
    Compute softmax function for input. 
    
    """
	### YOUR CODE HERE
    exp_scores = np.exp(x-np.amax(x,axis=1,keepdims=True))
    s = exp_scores/np.sum(exp_scores,axis=1,keepdims=True)
    s = np.clip(s,-1e-16,1e16)
    s = np.where(s>1e-16,s,1e-16)
	### END YOUR CODE
    return s

def sigmoid(x):
    """
    Compute the sigmoid function for the input.
    """
    ### YOUR CODE HERE
    s = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    ### END YOUR CODE
    return s

def rel_error(x, y):
    """ returns relative error """
    return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def net_loss(data,labels,params,
         reg = 0):
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    h = sigmoid(np.dot(data,W1)+b1)
    #h = np.maximum(0,a1) # ReLU activation
    y = softmax(np.dot(h,W2)+b2)
    corr_class_log_probs = np.log(y[np.where(labels==1)])
    loss = -(1/data.shape[0])* np.sum(corr_class_log_probs) + 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))
    grada2 = y - labels
    #backprop into W2 and b2
    gradW2 = np.dot(h.T,grada2)
    gradb2 = np.sum(grada2, axis=0,keepdims=True)
    gradh = np.dot(grada2,W2.T) 
    # #backprop to the ReLU non-linearity 
    # gradh[a1 <= 0]=0
    # grada1 = gradh 
    #backprop to the sigmoid non-linearity
    grada1 = np.multiply(gradh,h*(1-h))
    #backprop into W1 and b1 
    gradW1 = np.dot(data.T,grada1)
    gradb1 = np.sum(grada1,axis=0,keepdims=True)
    # #add regularization gradient contribution
    gradW2 +=  reg * W2
    gradW1 +=  reg * W1  
    #store gradient
    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2
    return loss, grad

def eval_numerical_gradient(f, x,grad,verbose=True, h=1e-6):
    """
    a naive implementation of numerical gradient of f at x
    - f should be a function that takes a single argument
    - x is the point (numpy array) to evaluate the gradient at
    """

    fx = f(x) # evaluate function value at original point
    grad = np.zeros_like(grad)
    # # iterate over all indexes in x
    # it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    # while not it.finished:

    #     # evaluate function at x+h
    #     ix = it.multi_index
    #     oldval = x[ix]
    #     x[ix] = oldval + h # increment by h
    #     fxph = f(x) # evalute f(x + h)
    #     x[ix] = oldval - h
    #     fxmh = f(x) # evaluate f(x - h)
    #     x[ix] = oldval # restore

    #     # compute the partial derivative with centered formula
    #     grad[ix] = (fxph - fxmh) / (2 * h) # the slope
    #     if verbose:
    #         print(ix, grad[ix])
    #     it.iternext() # step to next dimension
    oldval = x
    fxph = f(oldval + h)
    fxmh = f(oldval - h)
    grad = (fxph - fxmh) / (2 * h)
    return grad

def gradient_check(data,labels,params):
    """
    For each weight matrix, return the relative error between:
        - the numerical gradient evaluated at the current point with eval_numerical_gradient
        - the analytical gradient computed with backward_prop 
     
    """
    loss, grad = net_loss(data,labels,params)
    # these should all be less than 1e-8 or so
    for param_name in grad:
        f = lambda W: net_loss(data, labels, params)[0]
        param_grad_num = eval_numerical_gradient(f, params[param_name])
        print('%s max relative error: %e' % (param_name, rel_error(param_grad_num, grad[param_name])))
    return

def gradient_update(params,grad,learning_rate):
    """
    return parameters updated with "Vanilla" gradient descent
    """
    params['W1'] += -learning_rate * grad['W1']
    params['b1'] += -learning_rate * grad['b1']
    params['W2'] += -learning_rate * grad['W2']
    params['b2'] += -learning_rate * grad['b2']
    return params

def forward_prop(data, labels, params, reg):
    """
    return hidder layer, output(softmax) layer and loss
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    h = sigmoid(np.dot(data,W1)+b1)
    # h = np.maximum(0,a1) # ReLU activation
    y = softmax(np.dot(h,W2)+b2)
    #the properties of the softmax function and the cross entropy loss lead to a simple expression for the cost
    #it only depends on the log-probabilities of the classes whose true label is 1
    corr_class_log_probs = np.log(y[np.where(labels==1)])
    cost = -(1/data.shape[0])* np.sum(corr_class_log_probs) + 0.5*reg*(np.sum(W1*W1) + np.sum(W2*W2))
    return h, y, cost

def backward_prop(data, labels, params, reg):
    """
    return gradient of parameters
    """
    W1 = params['W1']
    b1 = params['b1']
    W2 = params['W2']
    b2 = params['b2']

    ### YOUR CODE HERE
    h, y, cost = forward_prop(data, labels, params,reg)
    #backprop into the cross entropy loss and softmax function
    grada2 = y - labels
    #backprop into W2 and b2
    gradW2 = np.dot(h.T,grada2)
    gradb2 = np.sum(grada2, axis=0,keepdims=True)
    gradh = np.dot(grada2,W2.T)
    # #backprop to the ReLU non-linearity 
    # gradh[a1 <= 0]=0
    # grada1 = gradh 
    #backprop to the sigmoid non-linearity
    grada1 = np.multiply(gradh,h*(1-h))
    #backprop into W1 and b1 
    gradW1 = np.dot(data.T,grada1)
    gradb1 = np.sum(grada1,axis=0,keepdims=True)
    #add regularization gradient contribution
    gradW2 +=  reg * W2
    gradW1 +=  reg * W1 
    #add batch normalization
    gradW1 = gradW1/data.shape[0]
    gradb1 = gradb1/data.shape[0]
    gradW2 = gradW2/data.shape[0]
    gradb2 = gradb2/data.shape[0]
    
    #store gradients
    grad = {}
    grad['W1'] = gradW1
    grad['W2'] = gradW2
    grad['b1'] = gradb1
    grad['b2'] = gradb2
    ### END YOUR CODE
    return grad

def visualize(train_loss_history, train_acc_history, dev_loss_history, dev_acc_history, grad_norm, diagnostics = True):
    x = np.arange(len(train_loss_history))
    #plot loss and accuracy at each epoch
    plt.figure(dpi=200)
    plt.plot(x,train_loss_history,label="training set")
    plt.plot(x,dev_loss_history,label="dev set")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.legend()
    plt.figure(dpi=200)
    plt.plot(x,train_acc_history,label="training set")
    plt.plot(x,dev_acc_history,label="dev set")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.legend()
    if diagnostics:
        plt.figure()
        plt.plot(np.arange(len(grad_norm)),grad_norm)
        plt.xlabel("Iteration")
        plt.ylabel("Gradient L2 Norm")
        plt.legend()
    return 

def nn_train(trainData, trainLabels, devData, devLabels,
             num_hidden     = 300,
             learning_rate  = 1,
             batch_size     = 1000, 
             num_epochs     = 30,
             reg            = 1,
             scale          = 1,
             lr_decay       = 1,
             load_weights   = False,
             save_weights   = True,
             verbose        = True,):
    (num_train,dim) = trainData.shape
    num_labels = trainLabels.shape[1]
    num_hidden = 300
    params = {}
    ### YOUR CODE HERE
    if load_weights:
        data = np.load("params.npz")
        params['W1'] = params['W1']
        params['b1'] = data['b1']
        params['W2'] = data['W2']
        params['b2'] = data['b2']
    else:      
        params['W1'] = np.random.randn(dim,num_hidden) * scale
        params['b1'] = np.zeros((1,num_hidden))
        params['W2'] = np.random.randn(num_hidden,num_labels)*scale
        params['b2'] = np.zeros((1,num_labels))
        
    iterations_per_epoch = max(int(num_train / batch_size),1)
    num_iters = int(iterations_per_epoch*num_epochs)
    train_loss_history,train_acc_history,dev_loss_history, dev_acc_history = [],[],[],[]
    grad_norm = []
    train_loss_epoch,train_acc_epoch = [],[]
    for it in range(num_iters):
        #generate a uniform random sample from np.arange(size of example set) of size batch_size
        train_mask = np.random.choice(np.arange(trainData.shape[0]),batch_size)
        #use that sample as indices to generate batches 
        trainData_b = trainData[train_mask]
        trainLabels_b = trainLabels[train_mask]
        
        #backprop
        grad = backward_prop(trainData_b, trainLabels_b, params, reg)
        
        #update the gradients
        gradient_update(params,grad,learning_rate)
        
        #compute cost & accuracy
        _, train_output_b, train_cost_b = forward_prop(trainData_b, trainLabels_b, params, reg)
        train_acc_b = compute_accuracy(train_output_b,trainLabels_b)
        
        grad_norm.append(np.linalg.norm(grad['W2']))
        train_loss_epoch.append(train_cost_b)
        train_acc_epoch.append(train_acc_b)
        
        #Every epoch, check loss and accuracy averaged over the entire training set and decay the learning rate
        if it % iterations_per_epoch == 0:
            #compute accuracy and loss on dev set
            _, dev_output, dev_cost = forward_prop(devData, devLabels, params, reg)
            dev_acc = compute_accuracy(dev_output,devLabels)
            
            #store loss and accuracy on training set
            train_loss_history.append(np.mean(train_loss_epoch))
            train_acc_history.append(np.mean(train_acc_epoch))
            #store accuracy and loss on dev set 
            dev_loss_history.append(dev_cost)
            dev_acc_history.append(dev_acc)
            
            #reset loss and accuracy lists for next epoch
            train_loss_epoch = []
            train_acc_epoch = []
            
            # Decay learning rate
            learning_rate *= lr_decay
            
            if verbose:
                p_epoch = "Current Epoch: " + str(it/iterations_per_epoch)
                p_progress  ="Training progress: " + "{:.2f}".format(100*(it/num_iters)) + str(" %")
                p_loss = "loss: " + str(train_cost_b)
                p_acc = "accuracy: " + str(train_acc_b)
                p_W1 = "W1 mean: {}, W1 std: {}".format(np.mean(params['W1']),np.std(params['W1']))
                p_W2 = "W2 mean: {}, W2 std: {}".format(np.mean(params['W2']),np.std(params['W2']))
                p_y = "y mean: {}, y std: {}".format(np.mean(train_output_b),np.std(train_output_b))
                output(p_epoch,p_progress,p_W1,p_W2,p_y,p_loss,p_acc)
                
    #store learned weights
    if save_weights:
        np.savez('data/params.npz', W1 = params["W1"], b1 = params["b1"], W2 = params["W2"], b2 = params["b2"])
        
    #plot loss, accuracy and gradient norm     
    visualize(train_loss_history,train_acc_history, dev_loss_history,dev_acc_history, grad_norm, diagnostics = False)   
    ### END YOUR CODE

    return params

def nn_test(data, labels, params):
    h, output, cost = forward_prop(data, labels, params, reg=0)
    accuracy = compute_accuracy(output, labels)
    return accuracy

def compute_accuracy(output, labels):
    accuracy = (np.argmax(output,axis=1) == np.argmax(labels,axis=1)).sum() * 1. / labels.shape[0]
    return accuracy

def one_hot_labels(labels):
    one_hot_labels = np.zeros((labels.shape[0], 10))
    one_hot_labels[np.arange(labels.shape[0]),labels.astype(int)] = 1
    return one_hot_labels

def main():
    np.random.seed(100)
    trainData, trainLabels, testData, testLabels = loadData()
    trainLabels = one_hot_labels(trainLabels)
    p = np.random.permutation(60000)
    trainData = trainData[p,:]
    trainLabels = trainLabels[p,:]
    devData = trainData[0:1000,:]
    devLabels = trainLabels[0:1000,:]
    trainData = trainData[1000:,:]
    trainLabels = trainLabels[1000:,:]
    
    mean = np.mean(trainData)
    std = np.std(trainData)
    trainData = (trainData - mean) / std
    devData = (devData - mean) / std
    
    testLabels = one_hot_labels(testLabels)
    testData = (testData - mean) / std
    #Check gradient
    # params_gc = nn_train(trainData, trainLabels, devData,
    #                       devLabels,num_epochs=1,num_hidden=1,verbose=True)
    # gradient_check(trainData,trainLabels,params_gc)
    params = nn_train(trainData, trainLabels, devData, devLabels)

    readyForTesting = True
    if readyForTesting:
          accuracy = nn_test(testData, testLabels, params)
          print('Test accuracy: %f' % accuracy)

if __name__ == '__main__':
    main()
