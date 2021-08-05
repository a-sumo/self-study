# -----------------------------------------------------------
# CS229: Machine Learning Assignment 1 
#
# author: Armand Sumo
# 
# email: armandsumo@gmail.com
# -----------------------------------------------------------

import numpy as np
import matplotlib.pyplot as plt
"""
Part 1: Logistic Regression
"""
x = np.loadtxt(open("data/logistic_x.txt","r")) #input array
y = np.loadtxt(open("data/logistic_y.txt","r")) #target array

m = x.shape[0]                                  #number of wavelengths
x = np.c_[np.ones(m),x]                         #append intercept term to x
y = np.expand_dims(y, axis=1)                   #reshape y for array broadcasting
theta = np.zeros((x.shape[1],1))                #theta: parameter array

def sigmoid_function(x):
    x = np.clip( x, -500, 500 )                 #prevent overflow
    x = 1.0/( 1 + np.exp( -x ))
    return x

def gradient_J(theta,x,y):
    #gradJ: gradient of the logistic cost function with respect to the parameter array theta
    a = np.multiply(y,x)
    b = 1-sigmoid_function(np.multiply(y,np.dot(x,theta)))
    grad_J = (-1/m)*np.sum(np.multiply(a,b),axis=0)
    if theta.shape[0] != grad_J.shape[0]:
        print("The dimensions of the gradient of the cost function J with respect to the vector theta are incorrect.")
    return grad_J

def hessian_J(theta,x,y):
    a = sigmoid_function(y*np.dot(x,theta))
    right_term = a*(1-a)
    left_term = np.einsum("ij,ik->ijk",x,x)
    H = np.einsum('ijk,il->jk',left_term,right_term)
    return H/m

def rel_error(x, y):
  """ returns relative error """
  return np.max(np.abs(x - y) / (np.maximum(1e-8, np.abs(x) + np.abs(y))))

def newton(theta,x,y,tolerance=0.01,itercount=False):
    niter=0
    while True:
        G = gradient_J(theta,x,y)
        H = hessian_J(theta,x,y)
        c = np.expand_dims(np.dot(np.linalg.inv(H),G),axis=1)
        theta_1 = theta - c
        if (rel_error(theta_1,theta)<tolerance):
            break
        theta =  theta_1
        niter+=1
    if itercount:
        print('Number of loop iterations:',niter)
    return theta
theta = newton(theta,x,y)
print("parameter vector theta for logistic regression:",theta)

#Plot Training data and decision boundary fit by logistic regression

#plt.title("Training data and decision boundary fit by logistic regression")
plt.figure(dpi=200)
plus = np.where(y>0)[0]
minus = np.where(y<0)[0]
plt.scatter(x[plus,1],x[plus,2],color = "r", marker="+",label="y= 1")
plt.scatter(x[minus,1],x[minus,2],color="b", marker="_",label="y= -1")
h = np.dot(x,theta)
plt.xlabel("x1")
plt.ylabel("x2")
def boundary(theta,x):
    a = -theta[1]/theta[2]
    b = -theta[0]/theta[2]
    return a*x + b 
# define bounds of the domain
min1, max1 = x[:, 1].min()-1, x[:, 1].max()+1
min2, max2 = x[:, 2].min()-1, x[:, 2].max()+1
#plot decision boundary
xbound = np.arange(min1,max1,0.1)
ybound = boundary(theta,xbound)
plt.plot(xbound,ybound,label="decision boundary")
plt.legend()

"""
Part 5: Regression for denoising quasar spectra
"""
# import numpy as np
# import matplotlib.pyplot as plt
import pandas as pd


# load quasar data for first training example 
data = pd.read_csv("D:/stanford-cs229/assigment 1/data/quasar_train.csv", nrows=2,header=None).to_numpy()
x1 = np.expand_dims(data[0],axis=1)
y1 = np.expand_dims(data[1],axis=1)

nlambda = x1.shape[0]                                                           #number of wavelength values
x1 = np.c_[np.ones(nlambda),x1]                                                 #append intercept term to x
theta1 = np.zeros((x1.shape[1],1))                                              #theta: parameter array

#first we do non-weighted linear regression
#the value of theta that minimizes our cost function is given by:
theta1 = np.dot(np.linalg.inv(x1.T@x1),x1.T@y1)

print("parameter vector theta for unweighted linear regression:",theta1)
#the straight line fit by non-weighted linear regression is given by:
y_nw = np.dot(x1,theta1)

#plot non-weighted linear regression raw and predicted output values
plt.figure(dpi=200)
plt.xlabel("Wavelength λ")
plt.ylabel("Flux")
plt.plot(x1[:,1],y1,color="c",label="raw y")
plt.plot(x1[:,1],y_nw,color="b",label="fit y",linewidth=1.2)
plt.legend()


#Weighted linear regression

tau = [1,5,10,100,1000]                                                            #bandwidth parameters
y_w = np.zeros((len(tau),x1.shape[0]))
for i in range(len(tau)):
    for j in range(nlambda):
        w_j = np.exp(-(x1[j,1]-x1[:,1])**2/(2*(tau[i])**2))                          
        W  = np.diag(w_j)                                                          #diagonal weight matrix
        theta2 = np.linalg.inv(np.dot(x1.T,W).dot(x1)).dot(np.dot(x1.T,W).dot(y1)) #theta2: optimal parameter array
        y_w[i,j] = np.squeeze(np.dot(theta2.T,x1[j,:]))

for i in range(len(tau)):
    plt.figure(dpi=200)
    plt.plot(x1[:,1],y1,color="c",label="raw y")
    plt.plot(x1[:,1],y_w[i,:],color="b",label="fit y with τ="+str(tau[i]),linewidth=1.2)
    plt.xlabel("Wavelength λ")
    plt.ylabel("Flux")
    plt.legend()

#Predicting quasar spectra with functional regression


# Load train and test quasar data
data_train = pd.read_csv("D:/stanford-cs229/assigment 1/data/quasar_train.csv",header=None).to_numpy()
data_test = pd.read_csv("D:/stanford-cs229/assigment 1/data/quasar_test.csv",header=None).to_numpy()
x2 = np.expand_dims(data_train[0],axis=1)
y_train = data_train[1:].T
f_train = y_train.copy()
y_test =  data_test[1:].T                                           #estimated regression functions y_train for test data
f_test = y_test.copy()                                              #estimated regression functions for train data

nlambda = f_train.shape[0]                                          #number of wavelength values
n_train = f_train.shape[1]                                          #number of training examples
n_test = f_test.shape[1]                                            #number of test examples    
x2 = np.c_[np.ones(nlambda),x2]                                          #append intercept term to x
theta_train = np.zeros((x2.shape[1],y_train.shape[1]))              #theta_train: optimal parameter array for train data
theta_test =  np.zeros((x2.shape[1],y_test.shape[1]))               #theta_test: optimal parameter array for test data


#Apply smoothing to the entire spectrum for both test and train data
tau_func_reg=5                                                      #bandwidth parameter for functional regression

for j in range(nlambda):
    w_j = np.exp(-(x2[j,1]-x2[:,1])**2/(2*(tau_func_reg)**2))
    W  = np.diag(w_j)                                               #diagonal weight matrix
    theta_train = np.linalg.inv(np.dot(x2.T,W).dot(x2)).dot(np.dot(x2.T,W).dot(y_train))
    theta_test = np.linalg.inv(np.dot(x2.T,W).dot(x2)).dot(np.dot(x2.T,W).dot(y_test))
    f_train[j,:] = np.squeeze(np.dot(theta_train.T,x2[j,:]))
    f_test[j,:] = np.dot(theta_test.T,x2[j,:])

def ker(t):
    if 1-t<0:
        return 0
    else:
        return t

def dist(f1,f2):
    #compute squared distance between new datapoint and previous data points
    return np.sum((f1-f2)**2,axis=0)

def knn(f,k,n_ex):
    #finds the k indexes that are closest to f_right using the metric defined in the "dist" function
    #returns: 
    #k_nearest_n: array containing the indexes of the k nearest neighbours for each training example
    #max_dist: array max_containing the maximum distances between each training example(not included) and all other training examples  
    k_nearest_n = np.zeros((n_ex,k))
    max_dist = np.zeros((n_ex,1))
    for i in range(n_ex):
        fright_i = np.expand_dims(f[:nlambda-299,i],axis=1)
        fright_not_i = np.delete(f[:nlambda-299,:],i,axis=1)
        dist_arr= dist(fright_i,fright_not_i)
        sorted_idx = np.argsort(dist_arr)
        k_nearest_n[i,:] = sorted_idx[-k:]
        max_dist[i] = dist_arr[sorted_idx[-1]]
    return (k_nearest_n,max_dist)

### Perform functional regression on the test set

k_nearest_n_train, max_dist_train = knn(f_train,3,n_train)[0],knn(f_train,3,n_train)[1]
f_left_train = f_train[:50,:]
f_left_hat_train = np.zeros_like(f_left_train)
error_train = np.zeros(n_train)
for i in range(n_train):
    sum_num_train = np.zeros(f_left_train.shape[0])
    sum_den_train = np.zeros(f_left_train.shape[0])
    for j in  k_nearest_n_train[i,:].astype(np.int32):
        fright_i_train = np.expand_dims(f_train[:nlambda-299,i],axis=1)
        fright_j_train = np.expand_dims(f_train[:nlambda-299,j],axis=1)
        sum_num_train = sum_num_train + ker(dist(fright_i_train,fright_j_train)/max_dist_train[i])*f_left_train[:,j]
        sum_den_train = sum_den_train + ker(dist(fright_i_train,fright_j_train)/max_dist_train[i])
    f_left_hat_train[:,i] = sum_num_train/sum_den_train
    error_train[j]=dist(f_left_train[:,i],f_left_hat_train[:,i])
#compute average error over the training data
average_error_train = np.average(error_train)
### Perform functional regression on the test set

k_nearest_n_test,max_dist_test = knn(f_test,3,n_test)[0],knn(f_test,3,n_test)[1]
f_left_test = f_test[:50,:]
f_left_hat_test = np.zeros_like(f_left_test)
error_test = np.zeros(n_train)
for i in range(n_test):
    sum_num_test = np.zeros(f_left_test.shape[0])
    sum_den_test = np.zeros(f_left_test.shape[0])
    for j in  k_nearest_n_test[i,:].astype(np.int32):
        fright_i_train = np.expand_dims(f_train[:nlambda-299,i],axis=1)
        fright_j_train = np.expand_dims(f_train[:nlambda-299,j],axis=1)
        sum_num_test = sum_num_test + ker(dist(fright_i_train,fright_j_train)/max_dist_test[i])*f_left_train[:,j]
        sum_den_test = sum_den_test + ker(dist(fright_i_train,fright_j_train)/max_dist_test[i])
    f_left_hat_test[:,i] = sum_num_test/sum_den_test
    error_test[j]=dist(f_left_test[:,i],f_left_hat_test[:,i])
#compute average error over the test data
average_error_test = np.average(error_test)

print("Average Train Error:",average_error_train)
print("Average Test Error:",average_error_test)

#Plot smoothed spectrum and fitted curve f_left_hat for test examples 1 and 6
for i in[0,5]:
    plt.figure(dpi=200)
    plt.plot(x1[:,1],f_test[:,i],color="b",label="Smoothed test data with τ = "+str(tau_func_reg),linewidth=1.2)
    plt.plot(x1[:50,1],f_left_hat_test[:,i],color="c",label="Estimated spectrum of f_left")
    plt.xlabel("Wavelength λ")
    plt.ylabel("Flux")
    #plt.title("Test example "+str(i+1))
    plt.legend()





