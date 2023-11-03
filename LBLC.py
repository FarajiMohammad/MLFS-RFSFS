import sklearn.metrics.pairwise
import numpy as np
def corr(Y):
    C = sklearn.metrics.pairwise.cosine_similarity(Y.T)
    Dc = np.diag(np.sum(C, axis=1))

    return C, Dc