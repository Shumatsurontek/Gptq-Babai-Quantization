import numpy as np
from .lattice_utils import cholesky_factor, inverse_cholesky

def gptq_babai_quantize(w, H, q_levels, clip=True):
    """
    GPTQ interpreted as Babai's Nearest Plane algorithm.
    Args:
        w: weight vector (d,)
        H: Hessian matrix (d, d), SPD
        q_levels: 1D array of quantization levels
        clip: whether to clip to min/max q_levels or not
    Returns:
        w_hat: quantized weight vector
    """
    R = cholesky_factor(H)
    z = R @ w
    d = len(w)
    q = np.zeros(d)
    for i in reversed(range(d)):
        ri = R[i, i]
        if ri == 0:
            q[i] = 0
            continue
        residual = z[i] - np.dot(R[i, i+1:], q[i+1:])
        q_i = np.round(residual / ri)
        if clip:
            q_i = np.clip(q_i, q_levels.min(), q_levels.max())
        q[i] = q_i
    w_hat = inverse_cholesky(R) @ q
    return w_hat
