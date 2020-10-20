from __future__ import print_function
import numpy as np
import sys

def get_image(i):
    return "./resources/car/car_data_set_images/",str(i+1)+".jpg"

def get_triplets_car():
    f = "./resources/car/ordinal_data_preprocessed/T_ALL_REDUCED.csv"
    
    triplets_raw = np.genfromtxt(f, delimiter=',',dtype='int') - 1
    # Statistics of the dataset
    n = np.amax(triplets_raw) + 1
    
    n_triplets_raw = triplets_raw.shape[0]

    triplets = np.zeros((n_triplets_raw*2,3), dtype=int)
    triplets[:n_triplets_raw,:] = np.hstack((triplets_raw[:,[1]],triplets_raw[:,[0]],triplets_raw[:,[2]]))
    triplets[n_triplets_raw:,:] = np.hstack((triplets_raw[:,[2]],triplets_raw[:,[0]],triplets_raw[:,[1]])) 
   
    triplets = np.unique(triplets, axis=0)

    n_triplets = triplets.shape[0]
    
    labels = np.zeros((n,),dtype=int)
    labels[[i-1 for i in [2,6,7,8,9,10,11,12,16,17,25,32,35,36,37,38,39,41,44,45,46,55,58,60]]] = 0
    labels[[i-1 for i in [15,19,20,28,40,42,47,48,49,50,51,52,54,56,59]]] = 1
    labels[[i-1 for i in [1,3,4,5,13,14,18,22,24,26,27,29,31,33,34,43,57]]] = 2
    labels[[i-1 for i in [21,23,30,53]]] = 3
    
    n_clusters = 4

    return triplets, n, n_triplets, labels, n_clusters

def get_quadruplets_car():
    triplets, n, n_triplets, labels, n_clusters = get_triplets_car()

    quadruplets = np.hstack((triplets[:,[0]],triplets[:,[1]],triplets[:,[0]],triplets[:,[2]]))
    
    return quadruplets, n, n_triplets, labels, n_clusters
        
if __name__ == '__main__':
    get_quadruplets_car()
