#!/usr/bin/env python

"""
Copyright 2014 Michael Wilber, Sam (Iljung) Kwak
"""

from glob import glob
import generate_all_triplets
from matplotlib import pylab as pl
from skimage.io import imread
import numpy as np
try:
    import tste
except:
    print "To generate an embedding, please download tste.py from https://github.com/ucsd-vision/tste-theano/blob/master/tste.py"
    import sys; sys.exit(1)

def show_images(X, imgs,scale=1,figsize=(10,10)):
    """ Show several images on a plot. X = 2D coordinate location, imgs = a list of images corresponding to those points. """
    fig,ax = pl.subplots(figsize=figsize)
    for i,(x,y) in enumerate(X):
        w = float(imgs[i].shape[1])
        h = float(imgs[i].shape[0])
        new_w = (w / max(w,h))
        new_h = (h / max(w,h))
        ax.imshow(imgs[i],
                  extent=(x,x+new_w*scale,
                          y,y+new_h*scale)
                  )
    ax.set_xlim((np.min(X[:,0])-scale, np.max(X[:,0])+scale))
    ax.set_ylim((np.min(X[:,1])-scale, np.max(X[:,1])+scale))
    ax.set_xticks([])
    ax.set_yticks([])
    return fig,ax


if __name__=="__main__":
    print "Loading triplet data..."
    triplets = list(generate_all_triplets.all_triplets(glob("raw-json/*.json")))

    print "Reading images..."
    all_images = list(set([img for triplet in triplets for img in triplet]))
    actual_images = map(imread, all_images)
    img_map = {i: filename for filename,i in enumerate(all_images)}

    triplets_ids = np.array([(img_map[a], img_map[b], img_map[c])
                             for a,b,c in triplets
                    ], dtype='i4')

    print "Generating embedding (please be patient, 1000 iterations)"
    embedding = tste.tste(triplets_ids, no_dims=2, verbose=True)
    fig,ax = show_images(embedding, actual_images)
    fig.savefig("embedding.png")
    pl.show()
