The Food100 Dataset
===================

FAIR USE STATEMENT:This dataset contains copyrighted material under
the educational fair use exemption to the U.S. copyright law.

What's included?
----------------

- 100 food Images from Yummly.com, handpicked to contain only one
  cuisine. Please see [our paper][arxivpaper] for details.
- A preprocessed .csv file containing all inferred triplets;
  `all-triplets.csv`. (Note that because we sampled with replacement,
  there may be duplicate triplets; the CSV file contains 250320
  triplets, 190376 of which are unique. This represents about 39% of
  the total `(1000)*(999)*(998)/2` possible triplets that we could
  sample.)
- The "raw" dataset: We ran 3 repetitions of several different grid
  sizes. Each repitition has at least 50 "HIT" tasks. Each HIT task
  has 10 "grid screens". Each grid screen can yield several triplets.
- Example code:
  - `generate_all_triplets.py` shows how to parse the raw JSON files.
  It outputs the `all-triplets.csv` file.
  - `show_triplet_embedding.py` shows how to graph the inferred
  triplets as a 2D embedding. (Dependencies: `tste.py`, which you can
  get from [github][github], `numpy`, `matplotlib`, and `scikit-image`)
- Ingredient annotations for most images with dish names, ingredient
  names and amounts, and "tastes"


The structure of the raw JSON files
-----------------------------------

Each JSON file is an experiment run. It contains at least 50 HITs:

    [
    {
        "worker_id": 308, // A unique ID representing the Mechanical Turk worker
        "HIT_screens": [
            {
                // One screen of the HIT is the smallest unit of human work
                // and represents the answer to one grid question.
                "images": [ // Set of all images shown to the user
                    "images/ca205931926143f0b0f4151971c58223.jpg",
                    "images/221eeb5fd8ab45778a07399eb192f3b1.jpg",
                    "images/6c51e12ae05e4c6da6652cc51fa258b6.jpg",
                    "images/89d550f0bd6a43638b4a0c85dcad644b.jpg",
                    ...
                 ]
                 // The "probe" (not contained in "images") is the
                 // image off to the left of the grid.
                "probe": "images/38aab5a720aa4b8780146c541ac84bb6.jpg",
                "near_answer": [ // Images the user selected
                    "images/6c51e12ae05e4c6da6652cc51fa258b6.jpg",
                    "images/3596b8da1ff84439b478f85b5ddc3e5b.jpg",
                    "images/00ca9e5230c749439a4b320e696e3819.jpg",
                    "images/27666b1933974c48af19b59ffbf3ab85.jpg"
                ],

                // For quality control, we include catch trials. We
                // did not need to filter results based on them, but
                // since we handpicked them, the example code omits
                // triplets inferred from catch trials.
                "is_catchtrial": true,

                // How long the user took to complete this grid question
                "timer": 33763
            },
            // ... (9 more grid screens in this hit)
        ]
    },
    // ... (50 more HITs in this experiment)
    ]


Ingredient information
----------------------

The `ingredients.json` file maps image filenames to structured
ingredient information:

    {'abcd.jpg': {
        'name': 'Tuscan White Beans with Spinach, Shrimp and Feta',
        'ingredients': [
           {
             'amount': '2 tbsp',
             'name': 'extra virgin olive oil',
           },
           ...
        ],
        'tastes': [
          {
            'percent': '33',
            'name': 'Savory'
          },
          ...
        ]
      },
      ...
    }

The following images are missing ingredient annotations:

- 6c51e12ae05e4c6da6652cc51fa258b6.jpg
- 0e37c510bb504d21baf495365c71ee16.jpg
- 6013a7400334423599602f61c50f5106.jpg
- 3a0b420b53d34f9aa8a9fdf50b43f757.jpg
- 38aab5a720aa4b8780146c541ac84bb6.jpg
- 221eeb5fd8ab45778a07399eb192f3b1.jpg
- 3e586e55aacc420fa39a25820c69ee1b.jpg

[github]: https://github.com/ucsd-vision/tste-theano/blob/master/tste.py
[arxivpaper]: http://arxiv.org/abs/1404.3291
