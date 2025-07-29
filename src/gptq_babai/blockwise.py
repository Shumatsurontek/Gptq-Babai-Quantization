import numpy as np
from .core import gptq_babai_quantize

def blockwise_quantize(w: np.ndarray, H: np.ndarray, q_levels: np.ndarray, block_size: int = 128, clip=True):
    """
    Quantifies vector w in blocks using GPTQ-Babai.
    Args:
        w: (d,) original weight vector
        H: (d, d) full Hessian
        q_levels: quantization levels (e.g. np.arange(-8, 9))
        block_size: size of blocks (e.g. 128)
        clip: whether to clip output
    Returns:
        w_hat: quantized vector, same shape as w
    """
    d = len(w)
    w_hat = np.zeros_like(w)
    for start in range(0, d, block_size):
        end = min(start + block_size, d)
        w_block = w[start:end]
        H_block = H[start:end, start:end]
        w_hat[start:end] = gptq_babai_quantize(w_block, H_block, q_levels, clip=clip)
    return w_hat
