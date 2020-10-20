#!/usr/bin/env python

"""
Copyright 2014 Michael Wilber, Sam (Iljung) Kwak
"""

import simplejson
from glob import glob

def all_triplets(filenames):
    """
    Yield all triplets that we could possibly infer
    """
    for file in filenames:
        HITs = simplejson.load(open(file))
        for hit in HITs:
            for screen in hit['HIT_screens']:
                if not screen["is_catchtrial"]:
                    all_images = set(screen["images"])
                    near = set(screen["near_answer"])
                    far = (all_images - near)

                    a = screen["probe"]
                    for b in near:
                        for c in far:
                            yield (a,b,c)


if __name__=="__main__":
    # Print all triplets:
    for (a,b,c) in all_triplets(glob("raw-json/*.json")):
        print "%s; %s; %s" % (a,b,c)

    # Print only unique triplets:
    """
    for (a,b,c) in set(all_triplets(glob("raw-json/*.json"))):
        print "%s; %s; %s" % (a,b,c)
    """
