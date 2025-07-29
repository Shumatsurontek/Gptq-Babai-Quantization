import torch
import numpy as np
from .blockwise import blockwise_quantize

def quantize_linear_layer(layer: torch.nn.Linear, hessian: np.ndarray, q_bits: int = 4, block_size: int = 128, clip=True):
    """
    Applies GPTQ-Babai quantization on a nn.Linear layer's weight.
    Args:
        layer: torch.nn.Linear layer
        hessian: approximated Hessian matrix of the layer (in np.ndarray)
        q_bits: number of bits (e.g. 4 → levels -8 to 7)
        block_size: block size for blockwise quant
        clip: whether to clip the quantized levels
    Returns:
        torch.Tensor: quantized weights
    """
    W = layer.weight.detach().cpu().numpy()  # shape (out, in)
    q_levels = np.arange(-2**(q_bits-1), 2**(q_bits-1))
    W_quant = np.zeros_like(W)
    for i in range(W.shape[0]):
        W_quant[i, :] = blockwise_quantize(W[i, :], hessian, q_levels, block_size, clip)
    return torch.tensor(W_quant, dtype=layer.weight.dtype)
