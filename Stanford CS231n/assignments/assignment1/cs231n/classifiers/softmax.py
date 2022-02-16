from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****


    #forward pass
    # num_dims = W.shape[0]
    # num_training=X.shape[0]
    # num_classes=W.shape[1]
    # scores = np.zeros((num_training,num_classes))
    # max_scores=np.zeros(num_training)
    # for i in range(num_training):
    #   scores[i] = X[i,:].dot(W)
    #   max_scores[i]=np.amax(scores[i])
    #   #-max_scores[i] to avoid numeric instability
    #   exp_scores = np.exp(scores[i]-max_scores[i])
    #   prob_scores = exp_scores/np.sum(exp_scores)
    #   for d in range(num_dims):
    #     for k in range(num_classes):
    #       if k == y[i]:
    #         dW[d,k]+=X.T[d,i]*(prob_scores[k]-1)
    #       else:
    #         dW[d,k]+=X.T[d,i]*prob_scores[k]
    #   loss+=-np.log(prob_scores[y[i]])


    # loss/=num_training
    # loss+=0.5*reg*(W*W)
    
    # dW/=num_training
    # dW+= reg*W
    num_classes = W.shape[1]
    num_train = X.shape[0]
    for i in xrange(num_train):
        scores = X[i].dot(W)
        correct_class = y[i]
        exp_scores = np.zeros_like(scores)
        row_sum = 0
        for j in xrange(num_classes):
            exp_scores[j] = np.exp(scores[j])
            row_sum += exp_scores[j]
        loss += -np.log(exp_scores[correct_class]/row_sum)
        #compute dW loops:
        for k in xrange(num_classes):
          if k != correct_class:
            dW[:,k] += exp_scores[k] / row_sum * X[i]
          else:
            dW[:,k] += (exp_scores[correct_class]/row_sum - 1) * X[i]
    loss /= num_train
    reg_loss = 0.5 * reg * np.sum(W**2)
    loss += reg_loss
    dW /= num_train
    dW += reg * W

    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    #forward pass
    num_train = X.shape[0]
    scores = X.dot(W)
    exp_scores = np.exp(scores)
    row_sum = exp_scores.sum(axis=1)
    row_sum = row_sum.reshape((num_train, 1))

    #compute loss
    norm_exp_scores = exp_scores / row_sum
    row_index = np.arange(num_train)
    data_loss = norm_exp_scores[row_index, y].sum()
    loss = data_loss / num_train + 0.5 * reg * np.sum(W*W)
    norm_exp_scores[row_index, y] -= 1

    dW = X.T.dot(norm_exp_scores)
    dW = dW/num_train + reg * W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
