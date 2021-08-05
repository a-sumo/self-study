# -----------------------------------------------------------
# CS229: Machine Learning Assignment 2
#
# author: Armand Sumo
# 
# email: armandsumo@gmail.com
# -----------------------------------------------------------

from __future__ import division
import numpy as np
import matplotlib.pyplot as plt
# -----------------------------------------------------------
#Logistic Regression
# -----------------------------------------------------------

try:
    xrange
except NameError:
    xrange = range

def add_intercept(X_):
    m, n = X_.shape
    X = np.zeros((m, n + 1))
    X[:, 0] = 1
    X[:, 1:] = X_
    return X

def load_data(filename):
    D = np.loadtxt(filename)
    Y = D[:, 0]
    X = D[:, 1:]
    return add_intercept(X), Y

def calc_grad(X, Y, theta):
    m, n = X.shape
    grad = np.zeros(theta.shape)

    margins = Y * X.dot(theta)
    probs = 1. / (1 + np.exp(margins))
    grad = -(1./m) * (X.T.dot(probs * Y))

    return grad

def sigmoid_function(x):
    x = np.clip( x, -500, 500 )                 #prevent overflow
    x = 1.0/( 1 + np.exp( -x ))
    return x

def separation(theta,X,Y):
    # this function checks whether or not the logistic regression model 
    # used to estimate theta achieves separation 
    y = Y.copy()
    #predict label with learned theta
    pred = (sigmoid_function(X.dot(theta))>=0.5).astype(int)
    y[np.where(Y==-1)]=0
    y_0 = np.where(y==0)[0]
    y_1 = np.where(y==1)[0]
    comparison_0 = pred[y_0] == y[y_0]
    comparison_1 = pred[y_1] == y[y_1]
    if comparison_0.all() and comparison_1.all():
        print("The model achieved complete separation.")
    elif comparison_0.all() or comparison_1.all():
        print("The model achieved quasicomplete separation.")
  
def logistic_regression(X, Y):
    m, n = X.shape
    theta = np.zeros(n)
    learning_rate = 10

    i = 0
    max_training_rounds = 100000
    fig, axes = plt.subplots(2, 2, figsize=(9, 6))
    axes = axes.ravel()
    diff = []
    grad_norm = []
    while True and i <= max_training_rounds:
        i += 1
        prev_theta = theta
        grad = calc_grad(X, Y, theta)
        theta = theta  - learning_rate * (grad)
        norm = np.linalg.norm(prev_theta - theta)
        grad_norm.append(np.linalg.norm(grad))
        diff.append(norm)
        if i % 10000 == 0:
        # if i % 1 == 0:
            print('Finished {0} iterations; Diff theta: {1}; theta: {2}; Grad: {3}'.format(
                i, norm, theta, grad))
            
            for j in range(m):
                if Y[j] == 1:
                    axes[int(i / 10000 - 1)].scatter(X[j][1], X[j][2], color='red',s=0.5)
                else:
                    axes[int(i / 10000 - 1)].scatter(X[j][1], X[j][2], color='blue',s=0.5)
            x = np.arange(0, 1, 0.1)
            axes[int(i/10000-1)].plot(x, -(theta[0] + theta[1] * x) / theta[2])
            
        # if norm < 1e-15:
        #     print('Converged in %d iterations' % i)
        #     break
    # plt.figure(dpi=200)    
    # plt.xlabel('Training Rounds')
    # plt.ylabel('Theta Difference')
    # plt.plot(np.arange(i)+1,np.array(diff))
    print("Last difference between current and previous theta : ",diff[-1])
    return theta
# -----------------------------------------------------------
#Naive Bayes
# -----------------------------------------------------------

### Bernouilli event model

def load_data_nb(filename):
    D = np.genfromtxt(filename,delimiter=',')
    Y = D[1:, -1]
    X = D[1:, 1:-1]
    return X,Y

def split_data(X,Y,train_test_split):
    n,m = X.shape
    n_train = np.floor(n * train_test_split).astype(int)
    X_train = X[:n_train]
    Y_train = Y[:n_train]
    X_test = X[n_train:]
    Y_test = Y[n_train:]
    return X_train,Y_train,X_test,Y_test
    
def mle_estimates_b(X,Y):
    # this function calculates the maximum likelihood estimates of the parameters necessary to 
    # compute the prediction p(y = 1 | x) under the Naive Bayes assumption. 
    n,m = X.shape
    spam_mail_wrd = np.zeros(m)
    nspam_mail_wrd = np.zeros(m)
    p_x_y1 = np.zeros(m)                #p(x | y = 1) 
    p_x_y0 = np.zeros(m)                #p(x | y = 0)
    
    for j in range(m):
        spam_mail_wrd[j] = np.count_nonzero(np.isin(np.where(X[:,j]==1)[0],np.where(Y==1)[0]))
        nspam_mail_wrd[j] = np.count_nonzero(np.isin(np.where(X[:,j]==1)[0],np.where(Y==0)[0]))
        
        p_x_y1[j] = (1 + spam_mail_wrd[j])/(2 + np.count_nonzero(Y))
        p_x_y0[j] = (1 + nspam_mail_wrd[j])/(2 + np.where(Y==0)[0].shape[0])
    p_y = (1/n)*np.count_nonzero(Y)     #p(y = 1)
    
    return(p_x_y1,p_x_y0,p_y)

def predict(x, y, p_x_y1, p_x_y0, p_y, print_features = False, print_result = False):
    features = np.where(x==1)[0]
    logp_y1_x = np.sum(np.log(p_x_y1[features]))+ np.log(p_y)
    logp_y0_x = np.sum(np.log(p_x_y0[features]))+ np.log(1-p_y)
    pred_spam = int(logp_y1_x >= logp_y0_x)
    return (pred_spam==y)

def evaluate(X, Y, p_x_y1, p_x_y0, p_y):
    n = X.shape[0]
    correct_pred = 0
    for k in range(n):
        correct_pred += predict(X[k,:], Y[k], p_x_y1, p_x_y0, p_y)
    accuracy = correct_pred/n
    print("Accuracy:",accuracy)
    return accuracy
    
def main():
    # Logistic regression
    
    # print('==== Training model on data set A ====')
    # Xa, Ya = load_data('data/data_a.txt')
    # theta_a = logistic_regression(Xa, Ya)
    # separation(theta_a,Xa,Ya)
    # print('\n==== Training model on data set B ====')
    # Xb, Yb = load_data('data/data_b.txt')
    # #Xb = 10*Xb
    # theta_b = logistic_regression(Xb, Yb)
    # separation(theta_b,Xb,Yb)
    # plt.show()
    # Bernouilli Naive Bayes
    print('==== Training model on emails dataset ====')
    splits = np.array([0.01,0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9])
    evals = np.zeros_like(splits)
    X,Y = load_data_nb('data/emails.csv')
    for i in range(splits.shape[0]):
        print('==== Training model on emails dataset with train/test split = ',splits[i])
        X_train,Y_train,X_test,Y_test = split_data(X,Y,splits[i])
        p_x_y1,p_x_y0,p_y = mle_estimates_b(X_train,Y_train)
        evals[i] = evaluate(X_test, Y_test, p_x_y1, p_x_y0, p_y)
    
    plt.figure(dpi=200)
    plt.xlabel("Train/Test split")
    plt.ylabel("Accuracy")
    plt.plot(splits,evals)
    plt.show()
if __name__ == '__main__':
    main()




