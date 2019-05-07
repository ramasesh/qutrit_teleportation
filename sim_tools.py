import scipy as sp
import numpy as np

def fidelity(M, U):
    """Calculate the fidelity between two density matricies as defined in
    https://journals.aps.org/pra/pdf/10.1103/PhysRevA.71.062310"""
    sqrt_m = np.matrix(sp.linalg.sqrtm(M))
    return np.trace(sp.linalg.sqrtm(np.matmul(sqrt_m, np.matmul(U, sqrt_m))))**2

def vec_to_dm(vec):
    """Perform the outer product on a vector and return a density matrix"""
    if 1 in np.shape(vec) or len(np.shape(vec))==1:
        if type(vec) == list:
            dm = np.dot(np.array(vec).reshape(-1, 1), np.conjugate(np.array(vec).reshape(1, -1)))
            return dm
        else:
            dm = np.dot(vec.reshape(-1, 1), np.conjugate(vec.reshape(1, -1)))
            return dm
    return vec

def partial_trace(dm, k=1, dim=None):
    """This was largely taken from https://github.com/gsagnol/picos with minor changes"""
    sz = dm.shape
    if dim is None:
        if sz[0] == sz[1] and (sz[0] ** 0.5) == int(sz[0] ** 0.5) and (sz[1] ** 0.5) == int(sz[1] ** 0.5):
            dim = (int(sz[0] ** 0.5), int(sz[1] ** 0.5))
        else:
            raise ValueError('The default parameter dim=None assumes X is a n**2 x n**2 matrix')

    # checks if dim is a list (or tuple) of lists (or tuples) of two integers each
    T = [list, tuple]
    if type(dim) in T and all([type(d) in T and len(d) == 2 for d in dim]) and all(
            [type(n) is int for d in dim for n in d]):
        dim = [d for d in zip(*dim)]
        pdim = np.product(dim[0]), np.product(dim[1])

    # if dim is a single list of integers we assume that no subsystem is rectangular
    elif type(dim) in [list, tuple] and all([type(n) is int for n in dim]):
        pdim = np.product(dim), np.product(dim)
        dim = (dim, dim)
    else:
        raise ValueError('Wrong dim variable')

    if len(dim[0]) != len(dim[1]):
        raise ValueError('Inconsistent number of subsystems, fix dim variable')

    if pdim[0] != sz[0] or pdim[1] != sz[1]:
        print(pdim, sz)
        raise ValueError('The product of the sub-dimensions does not match the size of X')

    if k > len(dim[0]) - 1:
        raise Exception('There is no k-th subsystem, fix k or dim variable')

    if dim[0][k] != dim[1][k]:
        raise ValueError('The dimensions of the subsystem to trace over don\'t match')

    dim_reduced = [list(d) for d in dim]
    del dim_reduced[0][k]
    del dim_reduced[1][k]
    dim_reduced = tuple(tuple(d) for d in dim_reduced)
    pdimred = tuple([np.product(d) for d in dim_reduced])

    fact = np.zeros((np.product(pdimred), np.product(pdim)), dtype='complex')

    for iii in itertools.product(*[range(i) for i in dim_reduced[0]]):
        for jjj in itertools.product(*[range(j) for j in dim_reduced[1]]):
            # element iii,jjj of the partial trace

            row = int(sum([iii[j] * np.product(dim_reduced[0][j + 1:]) for j in range(len(dim_reduced[0]))]))
            col = int(sum([jjj[j] * np.product(dim_reduced[1][j + 1:]) for j in range(len(dim_reduced[1]))]))
            # this corresponds to the element row,col in the matrix basis
            rowij = col * pdimred[0] + row
            # this corresponds to the elem rowij in vectorized form

            # computes the partial trace for iii,jjj
            for l in range(dim[0][k]):
                iili = list(iii)
                iili.insert(k, l)
                iili = tuple(iili)

                jjlj = list(jjj)
                jjlj.insert(k, l)
                jjlj = tuple(jjlj)

                row_l = int(sum([iili[j] * np.product(dim[0][j + 1:]) for j in range(len(dim[0]))]))
                col_l = int(sum([jjlj[j] * np.product(dim[1][j + 1:]) for j in range(len(dim[1]))]))

                colij_l = col_l * pdim[0] + row_l
                fact[int(rowij), int(colij_l)] = 1

    return np.dot(dm.reshape(-1), fact.T).reshape(pdimred[0], pdimred[1])

def vectorize_dm(density_matrix):
    
    assert len(density_matrix.shape) == 2
    assert density_matrix.shape[0] == density_matrix.shape[1]

    return np.array(density_matrix).T.reshape(-1)

def vectorized_dm_to_dm(vectorized_dm):
    dim_sys = int(np.floor(np.sqrt(vectorized_dm.shape[0])))

    assert dim_sys * dim_sys == vectorized_dm.shape[0]

    return vectorized_dm.reshape((dim_sys, dim_sys)).T

def random_pure_state(dim_sys=3):
    """ Samples a Haar-random pure state and returns it"""

    psi = np.random.randn(dim_sys)
    return psi/np.linalg.norm(psi, ord=2)

def rot_x(theta):
    """Rotation matrix around the x axis"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.matrix([[c, -1j * s], [-1j * s, c]])

def rot_y(theta):
    """Rotation matrix around the y axis"""
    c, s = np.cos(theta / 2), np.sin(theta / 2)
    return np.matrix([[c, -s], [s, c]])

def rot_z(theta):
    """Rotation matrix around the z axis"""
    exp_z = np.exp(1j * theta / 2.)
    exp_z_inv = np.exp(-1j * theta / 2.)
    return np.matrix([[exp_z_inv, 0], [0, exp_z]])

def bit_to_trit(m, subspace='GE'):
    """Embeds a single-qubit unitary in a 3X3 (qutrit) matrix"""
    out = np.eye(3, dtype='complex')
    
    if subspace in ['GE', 'EG']:
        out[0:2, 0:2] = m
    elif subpsace in ['EF', 'FE']:
        out[1:3, 1:3] = m
    elif subspace in ['GF', 'FG']:
        out[0,0] = m[0,0]
        out[0,2] = m[0,1]
        out[2,0] = m[1,0]
        out[2,2] = m[1,1]
    
    return out

