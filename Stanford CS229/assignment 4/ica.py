### Independent Components Analysis
###
### This program requires a working installation of:
###
### On Mac:
###     1. portaudio: On Mac: brew install portaudio
###     2. sounddevice: pip install sounddevice
###
### On windows:
###      pip install pyaudio sounddevice
###

import sounddevice as sd
import numpy as np
import matplotlib.pyplot as plt
Fs = 11025

def normalize(dat):
    return 0.99 * dat / np.max(np.abs(dat))

def load_data():
    mix = np.loadtxt('data/mix.dat')
    return mix

def play(vec):
    sd.play(vec, Fs, blocking=True)

def sigmoid(x):
    """
    Compute the sigmoid function for the input.
    """
    ### YOUR CODE HERE
    s = np.where(x >= 0, 1 / (1 + np.exp(-x)), np.exp(x) / (1 + np.exp(x)))
    ### END YOUR CODE
    return s

def visualize(grad_norm,step):
    gs = grad_norm[::step]
    plt.figure(dpi=200)
    plt.plot(np.arange(len(gs)),gs)
    plt.xlabel("Epoch")
    plt.ylabel("Gradient Norm")
    plt.show
    
def unmixer(X):
    M, N = X.shape
    W = np.eye(N)

    anneal = [0.1, 0.1, 0.1, 0.05, 0.05, 0.05, 0.02, 0.02, 0.01, 0.01,
              0.005, 0.005, 0.002, 0.002, 0.001, 0.001]
    print('Separating tracks ...')
    ######## Your code here ##########
    #expand anneleaning array
    anneal = np.array(anneal)
    anneal_arr = np.zeros((anneal.shape[0])*2)
    anneal_midpts = (anneal[1:] + anneal[:-1])/2
    anneal_midpts = np.append(anneal_midpts,0.01)
    anneal_arr[::2] = anneal_midpts
    anneal_arr[1::2] = anneal
    #Gradient Ascent 
    learning_rate = 1e-3
    #eps = 1e-2
    max_iter = anneal_arr.shape[0]-1
    diff = [1e10]
    grad_norm = [0]
    j=0
    #while j<max_iter and diff[-1]>eps:
    while j<max_iter :
        for i in range(M):
            idx = np.random.randint(M)
            grad = 1 - 2*sigmoid(np.dot(np.multiply(W.T,X[idx,:]),X[idx,:].T))+ np.linalg.inv(W.T)
            W += learning_rate * grad
            grad_norm.append(np.linalg.norm(grad))
            diff.append(np.linalg.norm(grad_norm[-1]-grad_norm[-2]))
        j+=1
        #anneal learning rate
        #learning_rate *= anneal[j]
        learning_rate *= anneal_arr[j]
        print("diff",diff[-1])
    visualize(grad_norm,M)
    ###################################
    return W

def unmix(X, W):
    S = np.zeros(X.shape)

    ######### Your code here ##########
    for i in range(X.shape[0]):   
        S[i:] = np.dot(W,X[i,:])
    ##################################
    return S

def main():
    X = normalize(load_data())
    """
    for i in range(X.shape[1]):
        print('Playing mixed track %d' % i)
        play(X[:, i])
    """
    W = unmixer(X)
    S = normalize(unmix(X, W))
    
    for i in range(S.shape[1]):
        print('Playing separated track %d' % i)
        play(S[:, i])
        
if __name__ == '__main__':
    main()
