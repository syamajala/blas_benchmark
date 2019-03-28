import sklearn.datasets as datasets
import scipy.io
import scipy.sparse
import scipy.linalg
import numpy as np
import time


def gen_matrix(filename='matrix.mtx', dim=1000):
    m = np.tril(datasets.make_spd_matrix(dim))
    m = scipy.sparse.coo_matrix(m)
    scipy.io.mmwrite(filename, m, symmetry='symmetric')


mat = scipy.io.mmread('matrix.mtx')
mat = mat.toarray()
mat = np.tril(mat)

start = time.time()
cmat = scipy.linalg.cholesky(mat, lower=True)
end = time.time()

print(end - start)

fmat = np.zeros(mat.shape)
with open('factored_matrix.mtx', 'r') as f:
    lines = f.readlines()
    lines = map(lambda line: line.strip().split(), lines)
    lines = list(map(lambda line: (int(line[0])-1, int(line[1])-1, float(line[2])), lines))
    for i, j, e in lines:
        fmat[i, j] = e

print(np.allclose(fmat, cmat, rtol=1e-04, atol=1e-04))


rmat = scipy.io.mmread('factored_matrix_rg.mtx')
rmat = rmat.toarray()
rmat = np.tril(rmat)

print(np.allclose(rmat, cmat, rtol=1e-04, atol=1e-04))
