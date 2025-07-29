import numpy as np
from numpy.linalg import cholesky, inv
from fpylll import IntegerMatrix, LLL

def cholesky_factor(H):
    """Compute upper triangular Cholesky factor R such that H = R.T @ R"""
    return cholesky(H).T

def inverse_cholesky(R):
    """Inverse of Cholesky upper-triangular factor"""
    return inv(R)

def lattice_reduce(H, scale=1000):
    """
    Apply LLL reduction to a scaled version of H
    Args:
        H: (d, d) positive-definite matrix
        scale: int, multiplier to convert to integer lattice
    Returns:
        Reduced matrix (approximate)
    """
    A = IntegerMatrix.from_matrix((H * scale).astype(int).tolist())
    LLL.reduction(A)
    return np.array(A) / scale

