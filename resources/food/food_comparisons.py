from __future__ import print_function
import numpy as np
import sys,os

def get_image(i):
    prefix = "./resources/food/images/"
    names = [img for img in os.listdir(prefix)]

    return prefix,names[i]

def get_triplets_food():
    # Get the data
    images = "./resources/food/images/"
    names = ["images/"+img for img in os.listdir(images)]
    f = "./resources/food/all-triplets.csv"
    n = len(names)
    
    triplets_raw = np.genfromtxt(f, delimiter=";",dtype='str')

    func_entry = np.vectorize(lambda x: names.index(x.strip()))

    triplets = func_entry(triplets_raw)

    triplets = np.unique(triplets, axis=0)
    
    n_triplets = triplets.shape[0]

    return triplets, n, n_triplets

def get_quadruplets_food():
    triplets, n, n_triplets = get_triplets_food()

    quadruplets = np.hstack((triplets[:,[0]],triplets[:,[1]],triplets[:,[0]],triplets[:,[2]]))
    
    return quadruplets, n, n_triplets
    
if __name__ == '__main__':
    get_quadruplets_food()
