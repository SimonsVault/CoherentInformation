import numpy as np
from scipy.optimize import minimize

from math import sqrt
from time import time
from basics import Multinomial

class ProductBasis():
    ''' Represents a basis of the symmetric subspace of n qudits using product states '''

    def __init__(self, n, d, dtype = np.complex128, seed = None, maxfun = 100):
        self.n, self.d, self.dtype, self.maxfun = n, d, dtype, maxfun
        
        self.k = Multinomial((n, d - 1)) # Dimension of symmetric subspace
        self.RandomState = np.random.RandomState(seed) # Seed RandomState to be used by this class

        self._ConstructBasis()

    def _ConstructBasis(self):
        n, d, k = self.n, self.d, self.k
       
        ## Init raw vectors: First the d obvious vectors and the k-d rest randomly
        raw_eye = np.eye(d, dtype = self.dtype)
        n_real_coeff = (k - d) *  d * 2
        start_real_coeff = self.RandomState.rand(n_real_coeff)

        # Optimizer target for raw vectors
        def TotalOverlap(real_coeff):
            raw_rest = real_coeff.reshape(k - d, d, 2).view(self.dtype)[..., 0]
            self.raw_vectors = np.concatenate((raw_eye, raw_rest), axis = 0)
            self.raw_sp =  np.dot(self.raw_vectors.conj(), self.raw_vectors.T) ** self.n
            return np.linalg.norm(self.raw_sp)  

        # Call the optimizer
        res = minimize(TotalOverlap, start_real_coeff, method = 'L-BFGS-B', bounds = [(-1.0, 1.0)] * n_real_coeff, options = {'maxfun':self.maxfun})
        print(res['nit'], res['nfev'])
        #TotalOverlap(start_real_coeff)
        
        ## Compute Gram-Schmidt coefficients
        self._GramSchmidt()
        
    def _GramSchmidt(self):
        k = self.k
        self.basis_coeff = np.zeros((k, k), dtype = self.dtype)

        for i in range(k):
            self.basis_coeff[i][i] = 1.0
            for j in range(i):
                sp = self.ScalarProduct(j, i)
                self.basis_coeff[i] -= sp * self.basis_coeff[j]

            # Normalize
            sp = self.ScalarProduct(i, i)
            self.basis_coeff[i] /= sqrt(np.real(sp))

    def ScalarProduct(self, i, j):
        ''' Compute scalar product of vectors i and j using current coefficients and precompuated values for the raw scalar product '''
        return np.linalg.multi_dot([np.conjugate(self.basis_coeff[i]), self.raw_sp, self.basis_coeff[j]])

    def TestOrthonormality(self):
        corr = np.zeros((self.k, self.k), dtype = self.dtype)

        for i in range(self.k):
            for j in range(self.k):
                corr[i, j] = self.ScalarProduct(i, j)

        error = np.linalg.norm(corr - np.eye(self.k))
        return error
        
if __name__ == '__main__':
    startt = time()
    
    PB = ProductBasis(8, 4)
    print(PB.TestOrthonormality())
    print(PB.basis_coeff[-1])
    
    print(round(time() - startt, 2))