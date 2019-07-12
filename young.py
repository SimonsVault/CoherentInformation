import numpy as np
import matplotlib.pyplot as plt

import cProfile

from sympy.combinatorics import Permutation
from itertools import product, permutations, combinations, chain
from functools import reduce
from math import factorial, sqrt, cos, sin, tan, radians, log
from time import time

from basics import Multinomial
from symmetric import kPartitions, kPartitionsOrdered
from minpol import Eta
from sparse import SparseVector, SparseONB, BlockSymmetricVector, BlockSymmetricONB

Permutation.print_cyclic = False

def multi_combinations(iterable, counts):
    ''' This function is to multinomial coeff. what combinations is to the binomial '''

    if len(counts) == 1:
        for x in combinations(iterable, counts[0]):
            yield x
    elif len(counts) > 1:
        for x in combinations(iterable, counts[0]):
            for y in multi_combinations(set(iterable) - set(x), counts[1:]):
                yield x + y

class StandardTableau():
    ''' Represents a standard tableaux and computes dimensions, Young symmetrizer and its range '''

    def __init__(self, *partition):
        self.partition = tuple(i for i in partition if i > 0) # Strip tailing zeros
        self.n = sum(partition)

        # Compute the entries in the rows and columns
        self.rows = []
        self.columns = [[] for _ in range(self.partition[0])]

        start = 0
        for x in self.partition:
            self.rows.append(list(range(start, start + x)))
            for i in range(x):
                self.columns[i].append(start + i)
            start += x

        # Permutations that helps setting up the column stabilizer
        self.column_reorderer = ~Permutation(list(chain(*self.columns)))

        # Calculate sym. dimension and product of all hook numbers
        self.boxes = [(i, j) for j, col in enumerate(self.columns) for i in range(len(col))]
        self.lastbox = len(self.boxes) - 1

        self.hookprod = self.HookProduct()
        self.sym_dim = factorial(self.n) // self.hookprod

    def __str__(self):
        ''' Basic ASCII-art representation of the tableau '''

        dlen = len(str(self.n - 1)) # Length of string of largest displayed number
        size = dlen + 2 # Size of each cell

        s = (' ' + '_' * size) * len(self.rows[0]) + '\n' # top line

        for row in self.rows:
            for x in row:
                s += '| ' + str(x) + ' ' * (dlen - len(str(x)) + 1) # cell contents and walls
            s += '|\n' + ('|' + '_' * size) * len(row) + '|\n' # bottom lines and walls

        return s[:-1] # Return without last newline

    def __repr__(self):
        return '<StandardTableau corresponding to partition {}>'.format(self.partition)

    def HookLength(self, i, j):
        ''' Returns the hook length of cell i, j (i vertical, j horizontal coordinate) '''
        return len(self.rows[i]) + len(self.columns[j]) - (i + j + 1)

    def HookProduct(self):
        ''' Compute the product of all hook lengths '''
        f = 1
        for i, j in self.boxes:
            f *= self.HookLength(i, j)
        return f

    def UnitDim(self, d):
        ''' Calculate the dimension of the unitary repr. with dimension d '''
        t = 1
        for i, j in self.boxes:
            t *= d + j - i
        t //= self.hookprod
        return t

    def RowIterator(self):
        ''' Yields all permutations that stabilize the rows '''
        for row_permutations in product(*(permutations(row) for row in self.rows)):
            yield Permutation(list(chain(*row_permutations)))

    def ColumnIterator(self):
        ''' Yields all permutations that stabilize the columns '''
        for column_permutations in product(*(permutations(column) for column in self.columns)):
            yield self.column_reorderer * Permutation(list(chain(*column_permutations)))

    def RowIteratorSpecial(self, state, d):
        ''' Yields all permutations for this row modulo ones that stabilize this state '''

        for row_permutations in product(*(self._RowCombinations(row, state, d) for row in self.rows)):
            yield ~Permutation(list(chain(*row_permutations)))

    def _RowCombinations(self, row, state, d):

        row_state = [state[i] for i in row]
        count = [row_state.count(i) for i in range(d)]
        yield from multi_combinations(row, count)

    def NonstandardStatesIterator(self, d, state = None, box_i = 0):

        if box_i == 0:
            state = [0] * self.n

        i, j = self.boxes[box_i]

        if i > 0:
            start = state[self.rows[i-1][j]] + 1
        else:
            start = 0

        if j > 0:
            start = max(start, state[self.rows[i][j-1]])

        for k in range(start, d):

            state[self.rows[i][j]] = k

            if box_i < self.lastbox:
                yield from self.NonstandardStatesIterator(d, state, box_i + 1)
            else:
                yield state

    def YoungBasis(self, d):

        d_powers = [d ** i for i in reversed(range(self.n))] # Used for converting basis -> index
        dim = self.UnitDim(d)

        basis = np.zeros((dim, d ** self.n))

        # Iterate over possible occupation numbers for each row separately
        for i, state in enumerate(self.NonstandardStatesIterator(d)):

            # Loop over permutations
            for r in self.RowIteratorSpecial(state, d):
                state1 = list(map(state.__getitem__, r))
                for c in self.ColumnIterator():
                    sgn = c.signature()

                    state2 = list(map(state1.__getitem__, c))
                    new_i = sum(map(int.__mul__, state2, d_powers))

                    basis[i, new_i] += sgn

            # Apply Gram-Schmid procedure
            for j in range(i):
                sp = np.dot(basis[i], basis[j])
                if sp != 0.0:
                    basis[i] -= sp * basis[j] # Substract component parallel to x

            norm = np.linalg.norm(basis[i])
            basis[i] /= norm # Normalize

        return basis

    def YoungBasisSparse(self, d):

        dim = self.UnitDim(d)
        vectors = [SparseVector(self.partition) for _ in range(dim)]

        # Iterate over possible occupation numbers for each row separately
        for i, state in enumerate(self.NonstandardStatesIterator(d)):

            # Loop over permutations
            for r in self.RowIteratorSpecial(state, d):
                state1 = tuple(map(state.__getitem__, r))
                for c in self.ColumnIterator():
                    sgn = c.signature()

                    state2 = tuple(map(state1.__getitem__, c))
                    vectors[i][state2] += sgn

        basis = SparseONB(vectors, name = str(self))
        return basis
        
    def YoungBasisBlockSymmetric(self, d):

        dim = self.UnitDim(d)
        vectors = [BlockSymmetricVector(self.partition) for _ in range(dim)]

        # Iterate over possible occupation numbers for each row separately
        for state_i, state in enumerate(self.NonstandardStatesIterator(d)):

            for c in self.ColumnIterator():
                sgn = c.signature()
                state1 = tuple(map(state.__getitem__, c))

                occupations = [0] * (d * len(self.rows))
                for row_i, row in enumerate(self.rows):
                    for x in row:
                        occupations[row_i * d + state1[x]] += 1
                      
                vectors[state_i][tuple(occupations)] += sgn

        basis = BlockSymmetricONB(vectors, d, len(self.rows), name = str(self))
        return basis

class YoungEntropy():
    ''' Collects bases for irreps and uses them to calculate the entropy of symmetric operators '''

    def __init__(self, n, d, verbose = False):
        self.n = n
        self.d = d

        if verbose: print('Initializing the diagonalizer...')

        # Iterate over all possible tableaus and compute bases
        self.bases = []
        for part in kPartitionsOrdered(n, d):
            part = tuple(reversed(part))
            tabl = StandardTableau(*part)
            if verbose: print(tabl)

            basis = tabl.YoungBasis(d)
            self.bases.append((basis, tabl.sym_dim))

    def __repr__(self):
        return '<YoungEntropy with n = {0}, d = {1} and {2} bases>'.format(self.n, self.d, len(self.bases))

    def __call__(self, rho):
        ''' Calculates entropy of operator rho '''

        H = 0.0

        for basis, mult in self.bases:
            rho_i = np.linalg.multi_dot([basis, rho, basis.T]) # Calculate operator in this subspace
            eigval = np.linalg.eigvalsh(rho_i) # eigvalsh!?
            H += sum(map(Eta, eigval)) * mult

        return H

class YoungBasesSparse():
    ''' Collects bases for irreps and uses them to calculate the entropy of symmetric operators '''

    def __init__(self, n, d, verbose = False):
        self.n = n
        self.d = d
        self.verbose = verbose

        if verbose: print('Initializing the young bases...')

        # Iterate over all possible tableaus and compute bases
        self.bases = []
        for part in kPartitionsOrdered(n, d):
            part = tuple(reversed(part))
            tabl = StandardTableau(*part)
            if verbose: print(tabl)

            basis = tabl.YoungBasisSparse(d)
            self.bases.append((basis, tabl.sym_dim))

        if verbose: print('Young bases ready.')

    def __repr__(self):
        return '<YoungBasesSparse with n = {0}, d = {1} and {2} bases>'.format(self.n, self.d, len(self.bases))

    def EntropyFromOuter(self, prefactors, operators):

        H = 0.0

        for basis, mult in self.bases:
            if self.verbose: print(basis.name)

            full_operators = basis.MatrixFromOuter(operators)
            rho = np.einsum('i, ijk -> jk', prefactors, full_operators)

            eigval = np.linalg.eigvalsh(rho)
            H += sum(map(Eta, eigval)) * mult

        return H

    def SpectrumFromOuter(self, prefactors, operators):

        spectrum = []

        for basis, mult in self.bases:
            if self.verbose: print(basis.name)

            full_operators = basis.MatrixFromOuter(operators)
            rho = np.einsum('i, ijk -> jk', prefactors, full_operators)

            eigval = np.linalg.eigvalsh(rho)
            spectrum.extend(eigval * mult)

        return spectrum

def test_listdimensions(n, d):
    ''' List all young diagrams and the corresponding dimension of irreps '''

    s = 0
    s2 = 0
    for part in kPartitionsOrdered(n, d):
        part = tuple(reversed(part))
        tabl = StandardTableau(*part)
        sym_d, unit_d = tabl.sym_dim, tabl.UnitDim(d)
        s += sym_d * unit_d
        s2 += unit_d ** 2
        print(tabl)
        print('Sym. dim. {0} and unit. dim. {1}'.format(sym_d, unit_d))
    assert(s == d ** n)
    print(s2)

def test_youngbasis(n, d):
    ''' Collect all bases and check orthonormality '''

    basis_collection = None

    for part in kPartitionsOrdered(n, d):
        part = tuple(reversed(part))
        tabl = StandardTableau(*part)
        print(tabl)

        basis = tabl.YoungBasis(d)
        if basis_collection is None:
            basis_collection = basis
        else:
            basis_collection = np.vstack((basis_collection, basis))

    dim = basis_collection.shape[0]
    corr = np.zeros((dim, dim))
    for i in range(dim):
        for j in range(i + 1):
            corr[i,j] = np.dot(basis_collection[i], basis_collection[j])

    #print(corr)
    diff = np.linalg.norm(corr - np.eye(dim))
    assert(diff < 10 ** -10)
    print(diff)

def test_youngbasissparse(n, d):
    ''' Collect all bases and check orthonormality '''

    for part in kPartitionsOrdered(n, d):
        part = tuple(reversed(part))
        tabl = StandardTableau(*part)
        print(tabl)

        basis = tabl.YoungBasisBlockSymmetric(d)
        print('Basis created, checking...')
        print(basis.TestOrthonormality())

def operators_rochus1(n, d, alpha):

    op_num = 3
    prefactors = np.zeros(op_num)
    operators = np.zeros((op_num, d, d))

    prefactors[0] = 0.5
    operators[0, 0, 0] = cos(alpha) ** 2
    operators[0, 1, 1] = sin(alpha) ** 2

    prefactors[1] = 0.5
    operators[1, 0, 0] = sin(alpha) ** 2
    operators[1, 1, 1] = cos(alpha) ** 2

    prefactors[2] = (sin(alpha) * cos(alpha)) ** n
    operators[2, 0, 0] = 1.0
    operators[2, 1, 1] = -1.0

    return prefactors, operators

def operators_rochus2(n, d, alpha):

    op_num = 3
    prefactors = np.zeros(op_num)
    operators = np.zeros((op_num, d, d))

    prefactors[0] = 0.5
    operators[0, 0, 0] = cos(alpha) ** 2
    operators[0, 1, 1] = sin(alpha) ** 2

    prefactors[1] = 0.5
    operators[1, 0, 0] = operators[1, 1, 1] = 0.5
    operators[1, 1, 0] = operators[1, 0, 1] = 0.5 * (1.0 - 2.0 * cos(alpha) ** 2)

    prefactors[2] = (sin(alpha) * cos(alpha)) ** n
    operators[2, 0, 0] = operators[2, 1, 0] = operators[2, 0, 1] = 1.0 / sqrt(2.0)
    operators[2, 1, 1] = -1.0 / sqrt(2.0)

    return prefactors, operators

def entropy_check(n, d, alpha):

    # Compare to analytical result
    Han = 0.0
    for w in range(n + 1):
        ev = 0.5 * (cos(alpha)) ** (2 * n) * (tan(alpha) ** w + (-1) ** w * tan(alpha) ** (n-w)) ** 2
        Han += Multinomial((n - w, w)) * Eta(ev)

    return Han

def plot_outerop_entropy(n, d):

    bases = YoungBasesSparse(n, d, verbose = True)

    # Data for plotting
    x = np.arange(0.0, 90, 1)
    y_list = []
    z_list = []

    for alpha in x:
        prefac, op = operators_rochus1(n, d, radians(alpha))
        entropy = bases.EntropyFromOuter(prefac, op)
        y_list.append(entropy)

        entropy2 = entropy_check(n, d, radians(alpha))
        assert(abs(entropy - entropy2) < 10 ** - 10)

    y = np.array(y_list)
    z = np.array(z_list)

    fig, ax = plt.subplots()
    ax.plot(x, y, z)

    ax.set(xlabel=r'$\alpha$ (degrees)', ylabel='Entropy (base 2)')
    ax.grid()

    fig.savefig("entropy {}.png".format(n))
    plt.show()

def plot_outerop_spectrum(n, d, alpha):

    bases = YoungBasesSparse(n, d, verbose = True)

    prefac, op = operators_rochus1(n, d, radians(alpha))
    spectrum1 = bases.SpectrumFromOuter(prefac, op)

    prefac, op = operators_rochus2(n, d, radians(alpha))
    spectrum2 = bases.SpectrumFromOuter(prefac, op)

    data = []

    for i, s in enumerate([spectrum1, spectrum2]):

        spectrum = [0.0] + [ev for ev in sorted(s) if abs(ev) > 10 ** -10]
        print(spectrum)

        cum = 0.0
        cum_list = []

        for ev in sorted(spectrum):
            cum += ev
            cum_list.append(cum)

        data.append(spectrum)

    x = np.array(range(len(data[0])))
    y = np.array(data[0])
    z = np.array(data[1])

    fig, ax = plt.subplots()
    ax.plot(x, y, z)

    ax.set(xlabel = r'$\alpha$ (degrees)', ylabel = 'Entropy (base 2)')
    ax.grid()

    fig.savefig("spectrum {}.png".format(n))
    plt.show()

if __name__ == '__main__':

    n, d = 20, 4
    '''tabl = StandardTableau(n // 2, n // 2)
    print(tabl)

    prefactors, operators = operators_rochus1(n, d, radians(45))

    basis1 = tabl.YoungBasisSparse(d)
    basis2 = tabl.YoungBasisBlockSymmetric(d)

    t0 = time()
    matrix1 = basis1.MatrixFromOuter(operators)
    t1 = time()
    matrix2 = basis2.MatrixFromOuter(operators)
    #cProfile.run('matrix2 = basis2.MatrixFromOuter(operators)')
    t2 = time()
    print(round(t1-t0, 2),round(t2-t1, 2))

    print(np.linalg.norm(matrix1-matrix2))'''
    test_listdimensions(n, d)
