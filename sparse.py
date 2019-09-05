import numpy as np
import operator as op

from functools import reduce
from itertools import product
from math import sqrt, factorial

from basics import Multinomial_TM
from symmetric import kPartitions, TMatrices

class SparseVector():
    ''' Implements basic vector operations based on a dictionary of entries '''

    def __init__(self, coeff_dict = None):
        ''' Initialize using optionally given dictionary of coefficients'''
        self.dict = coeff_dict if type(coeff_dict) is dict else {}

    def __repr__(self):
        return 'SparseVector({})'.format(self.dict)

    def __getitem__(self, key):
        return self.dict[key] if key in self.dict else 0.0

    def __setitem__(self, key, item):
        self.dict[key] = item

    def __add__(self, other):
        ''' Implement addition '''

        result = type(self)()
        for key in self.dict.keys() | other.dict.keys(): # Coefficients in the union
            result[key] = self[key] + other[key]

        return result

    def __mul__(self, other):
        ''' Implement scalar product and scalar multiplication '''

        if isinstance(other, SparseVector): # Dot product

            # Sum over keys in the intersection
            result = sum(self.dict[key] * other.dict[key] for key in self.dict.keys() & other.dict.keys())

        else: # Let's try to use 'other' as a numeric object

            result = SparseVector()
            for key in self.dict.keys():
                result.dict[key] = self.dict[key] * other

        return result

class BlockSymmetricVector(SparseVector):
    ''' Implements basic vector operations based on a dictionary of entries '''

    def __init__(self, partition, coeff_dict = None):
        super().__init__(coeff_dict)

        self.partition = partition
        self.part_factorials = np.prod([factorial(x) for x in partition])

    def __mul__(self, other):
        ''' Implement scalar product and scalar multiplication '''

        if isinstance(other, BlockSymmetricVector): # Dot product
            assert(self.partition == other.partition)

            # Sum over keys in the intersection
            result = 0.0
            for key in self.dict.keys() & other.dict.keys():
                result += self.part_factorials / np.prod([factorial(x) for x in key]) * self.dict[key] * other.dict[key]
                
        else: # Let's try to use 'other' as a numeric object

            result = BlockSymmetricVector(self.partition)
            for key in self.dict.keys():
                result.dict[key] = self.dict[key] * other
            
        return result

class ONB():
    ''' Represents an ONB of vectors by implicitly using Gram-Schmid coefficients '''

    def __init__(self, vectors, name = ''):

        self.vectors = vectors
        self.dim = len(vectors)
        self.name = name
        
        # Precompute all the scalar products
        self.sp = np.zeros((self.dim, self.dim))

        for i in range(self.dim):
            for j in range(i + 1):
                self.sp[i, j] = self.sp[j, i] = vectors[i] * vectors[j]

        # Compute matrix coefficients of ONB using Gram-Schmidt
        self.coeff = np.zeros((self.dim, self.dim))

        for i in range(self.dim):
            self.coeff[i][i] = 1.0
            for j in range(i):
                sp = self.ScalarProduct(i, j)
                self.coeff[i] -= sp * self.coeff[j]

            self.Normalize(i)

    def __str__(self):
        return self.name

    def ScalarProduct(self, i, j):
        ''' Calculate scalar product of vectors indexed by i and j '''
        return np.linalg.multi_dot([self.coeff[i], self.sp, self.coeff[j]])

    def Normalize(self, i):
        ''' Normalize the vector indexed by i '''

        norm = np.linalg.multi_dot([self.coeff[i], self.sp, self.coeff[i]])
        self.coeff[i] /= sqrt(norm)

    def __getitem__(self, i):
        ''' Return vector i of ONB (mainly for use in TestOrthonormality) '''

        # Use reduce/op because sum uses initializer 0
        result = reduce(op.add, (self.vectors[j] * self.coeff[i, j] for j in range(self.dim)))
        return result

    def TestOrthonormality(self):
        ''' Calculates matrix of scalar products and returns difference from identity matrix '''

        corr = np.zeros((self.dim, self.dim))

        for i in range(self.dim):
            for j in range(self.dim):
                corr[i,j] = self[i] * self[j]

        return np.linalg.norm(corr - np.eye(self.dim))

class SparseONB(ONB):

    def __init__(self, vectors, name = ''):
        super().__init__(vectors, name)    

        # Create list that works with numba  
        self.vectors_numba = [(i, k, v) for i, vector in enumerate(vectors) for k, v in vector.dict.items()]

    def MatrixFromTM(self, tm):
        d = tm_operator.container.d

        vector_repr = np.zeros((self.dim, self.dim), dtype = np.complex128)

        for i in range(self.dim):
            for key_i in self.vectors[i].dict.keys():

                for j in range(i + 1):
                    for key_j in self.vectors[j].dict.keys():

                        tm = np.zeros((d, d), dtype = np.int64)

                        for k in range(n):
                            tm[key_i[k], key_j[k]] += 1

                        tm_i, _, _, tm_mult = tm_operator.container.dctTM[tm.tostring()]
                        s = self.vectors[i][key_i] * self.vectors[j][key_j] * tm_operator[tm_i] / sqrt(tm_mult)
                        vector_repr[i, j] += s
                        if i != j:
                            vector_repr[j, i] += s.conjugate()

        # Second step: Linear transform to represention in ONB using the computed coefficients
        return np.linalg.multi_dot((self.coeff, vector_repr, self.coeff.T))

    def MatrixFromOuter(self, operators):
    
        # Call numba'd subroutine
        vector_repr = self._NumbaMatrixFromOuter(self.dim, self.vectors_numba, operators)
        
        # Linear transform to represention in ONB using the computed coefficients
        return np.einsum('ij, ojk, lk -> oil', self.coeff, vector_repr, self.coeff)

class BlockSymmetricONB(ONB):

    def __init__(self, vectors, d, rows, name = ''):
        self.d, self.rows = d, rows
        super().__init__(vectors, name)

    def MatrixFromOuter(self, operators):

        # First step: Compute <vi|A|vj> for vi in self.vectors
        num_op = operators.shape[0]
        d, rows = self.d, self.rows
        key_blocks = [(row * d, (row + 1) * d) for row in range(self.rows)] 
        vector_repr = np.zeros((num_op, self.dim, self.dim), dtype = np.complex128)

        for i in range(self.dim):
            for key_i in self.vectors[i].dict.keys():

                for j in range(i + 1):
                    for key_j in self.vectors[j].dict.keys():

                        for tms_raw in product(*(TMatrices(key_i[s:e], key_j[s:e]) for s, e in key_blocks)):
                            tms = np.array(tms_raw).reshape(rows, d, d)
                            tm_sum = tms.sum(axis = 0)

                            mult = 1
                            for tm in tms:
                                mult *= Multinomial_TM(tm)

                            s = np.ones(num_op, dtype = np.complex128)

                            for x in range(d):
                                for y in range(d):
                                    s *= operators[..., x, y] ** tm_sum[x, y]

                            s *= self.vectors[i][key_i].conjugate() * self.vectors[j][key_j] * mult
                            vector_repr[..., i, j] += s
                            if i != j:
                                vector_repr[..., j, i] += s.conjugate()

        # Second step: Linear transform to represention in ONB using the computed coefficients
        return np.einsum('ij, ojk, lk -> oil', self.coeff, vector_repr, self.coeff)

def Test_SparseONB():

    n, dim = 20, 20

    # Prepare n random "sparse" vectors with dim entries
    vectors = [SparseVector() for _ in range(n)]
    rando = np.random.rand(n, dim)

    for i in range(n):
        for j in range(dim):
            vectors[i][j] = rando[i, j]

    # Create ONB and check if it's actually one
    onb = SparseONB(vectors)
    diff = onb.TestOrthonormality()
    print(diff)
    assert(diff < 10 ** -10)

if __name__ == '__main__':
    Test_SparseONB()
