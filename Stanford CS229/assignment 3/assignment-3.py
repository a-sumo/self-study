# -----------------------------------------------------------
# CS229: Machine Learning Assignment 3, Deep Learning & Unsupervised Learning
#
# author: Armand Sumo
# 
# email: armandsumo@gmail.com
# -----------------------------------------------------------
import numpy as np
np.seterr(divide='ignore', invalid='ignore')
import matplotlib.pyplot as plt
from matplotlib.image import imread

def load_data(filename):
    return imread(filename)

def k_means(A,k,eps,iter,display_progress=False):
    
    np.random.seed(iter)                    #make index values deterministic for reproducibility
    (m,n,p) = A.shape
    idx = np.random.randint(m,size=(k,k))
    distance = np.zeros((m,n,k))            #points/centroids distance array
    centroids  =  A[idx[:,0],idx[0,:],:]    #initialize cluster centroids randomly
    clust_A = A.copy()                      #array containing the rgb-spsace position of the cluster centroid assigned to each point
    distortion = 0                          #measure of the rbg space distance between points and assigned cluster centroids
    diff = [1e10]                           #difference between the distortions for two consecutive iterations
    i=0
    while abs(diff[-1])>eps:

        #assign each example to its closest centroid
        for j in range(k):
            distance[:,:,j] = np.linalg.norm(A-centroids[j],axis=2)
        ast = distance.argmin(axis=2)       #points/centroid assignment matrix
        
        #move centroid to the average position (in rgb-space) of the points to which it is assigned
        for j in range(k):
           ast_j = np.where(ast==j)
           centroids[j] = np.nan_to_num(np.sum(A[ast_j[0],ast_j[1]],axis=0))/np.nan_to_num(len(ast_j[0]))
           clust_A[ast_j] = centroids[j]
           
        distortion_new = np.linalg.norm(A-clust_A)
        diff.append(distortion - distortion_new)
        distortion = distortion_new
        if display_progress and i%100==0:
            print(" Current iteration:{} \n Distortion:\n{} Difference from previous iteration:\n{}, ".format(i,distortion,diff[-1]))
        i +=1     
            
    return clust_A, distortion
def main():
    A = load_data('data/mandrill-small.tiff')
    plt.imshow(A)
    k = 16                                  # number of clusters
    eps = 1                                 #convergence threshold
    iter = 2                                #number of k-means iterations
    distortion = np.zeros(iter+1)
    best_c = np.zeros_like(A)               #clustered array with the lowest distortion
    
    #iterate with different centroid initializations (avoid local minima)
    for i in range(iter):
        clust_A, distortion[i+1] = k_means(A,k,eps,iter,display_progress=True)
        print(distortion[-1])
        if abs(distortion[-1])< abs(distortion[-2]):
            best_c = clust_A
    plt.imshow(best_c)
    plt.show()
if __name__ == '__main__':
    main()