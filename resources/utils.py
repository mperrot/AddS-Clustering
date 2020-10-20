import numpy as np
import itertools
import scipy.stats


def planted_model(N, k, delta, distribution='NormalNormal'):
    """
    Generates similarity matrix with respect to two given classes.

    Parameters
    ----------
    N : int
        number of Datapoints

    k : int
        number of classes

    distribution : str
        F_in and F_out
        options:    'NormalNormal'
                    'UniformUniform'
                    'BetaBeta'
                    'BetaUniform'

    mu : float
        mean for the gaussian that is sampled for the similarity
        between classes

    sigma : float:
        std for the gaussian

    delat : float:
        shift of the mean if two indeces are not in the same cluster

    Returns
    -------
    w : NxN numpy array
        similarity matrix

    indes: list of length N
        list of cluster assignements

    cluster: list (k x (N/k))
        list of list with clusters
    """

    # define initial w
    w = np.zeros((N, N))

    num_samples = (N * (N + 1)) // 2

    if distribution == 'NormalNormal':

        sigma = 0.1
        mu_out = 0
        mu_in = np.sqrt(2) * sigma * scipy.stats.norm.ppf((1 + delta) / 2)

        # sample similarity values
        F_in = np.random.normal(mu_in, sigma, num_samples)
        F_out = np.random.normal(mu_out, sigma, num_samples)

    elif distribution == 'UniformUniform':

        a = 1 - np.sqrt(1 - delta)
        F_in = np.random.uniform(0 + a, 1 + a, num_samples)
        F_out = np.random.uniform(0, 1, num_samples)

    elif distribution == 'BetaBeta':

        b = 2
        a = b * ((1 + delta) / (1 - delta))
        F_in = np.random.beta(a, b, num_samples)
        F_out = np.random.beta(1, 1, num_samples)


    elif distribution == 'UniformNormal':

        mu = (1 + delta) / 2
        F_in = np.random.normal(mu, 1, num_samples)
        F_out = np.random.uniform(0, 1, num_samples)

    # create clusters
    cluster = np.array_split(np.arange(N), k)
    for i in range(k):
        cluster[i] = [i] * cluster[i].shape[0]

    cluster = list(itertools.chain(*cluster))

    # create similarity matrix
    n = 0
    m = 0
    for i in range(N):
        for j in range(i + 1, N):

            # check if the they are in the same
            if cluster[i] == cluster[j]:
                w[i, j] = F_in[n]
                w[j, i] = F_in[n]
                n += 1
            else:
                w[i, j] = F_out[m]
                w[j, i] = F_out[m]
                m += 1

    np.fill_diagonal(w, float("inf"))

    return w, cluster
