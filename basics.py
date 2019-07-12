import numpy as np
from math import log

def Eta(x):
    ''' Implements the function eta(x) = - x * ln(x) with eta(0) = 0 '''

    if x > 0:
        return -x * log(x, 2)
    else:
        return 0.0

def Multinomial(lst):
    # Source: https://stackoverflow.com/questions/46374185/does-python-have-a-function-which-computes-multinomial-coefficients

    res, i = 1, 1
    for a in lst:
        for j in range(1, a + 1):
            res *= i
            res //= j
            i += 1
    return res

def Multinomial_NP(array):
    # Adapted from Multinomial

    res, i = 1, 1
    for a in np.nditer(array):
        for j in range(1, a + 1):
            res *= i
            res //= j
            i += 1
    return res

def kPartitions(n, k):

    if k > 1:
        for i in range(n + 1):
            for p in kPartitions(n - i, k - 1):
                yield p + (i,)
    else:
        yield (n,)
