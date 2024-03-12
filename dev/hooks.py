import torch
from typing import Callable

def zero_ablate(index: int) -> Callable:
    """
    Returns ablation hook function which zero-ablates attention/mlp
    at the final token.

    Parameters
    ----------
    heads : required, list
        List of attention heads to ablate.
    index : required, int
        Position of the last token.

    Returns
    ------
    hook : Callable
        Ablation hook function.
    """

    def hook(model, input, output):
        output[0, index] = 0
        return output

    return hook

def zero_ablate_heads(heads: list, index: int) -> Callable:
    """
    Returns abblation hook function which zero-ablates |heads| at the
    final token.

    Parameters
    ----------
    heads : required, list
        List of attention heads to ablate.
    index : required, int
        Position of the last token.

    Returns
    ------
    hook : Callable
        Ablation hook function.
    """

    def hook(model, input, output):
        for h in heads:
            output[:, index, h * (256) : (h + 1) * 256] = 0
        return output

    return hook