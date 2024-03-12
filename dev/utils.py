import torch
import numpy as np

def to_numpy(t: torch.Tensor) -> np.ndarray:
    """
    Detach tensor from computational graph, place it on the cpu, and convert it
    to a numpy array.

    Parameters
    ----------
    t : tensor, required
        Any pytorch tensor.

    Returns
    ------
    np.ndarray
        Converted numpy array.
    """
    return t.detach().cpu().numpy()

def mean_up_low(t: torch.Tensor) -> (np.ndarray, np.ndarray, np.ndarray):
    """
    Compute the mean, upper bound on confidence, and lower bound on confidence
    of the values.

    Parameters
    ----------
    t : torch.Tensor, required
        Tensor with numerical values.

    Returns
    ------
    t_mean : np.ndarray
        Mean values.
    t_up : np.ndarray
        Upper confidence values.
    t_low : np.ndarray
        Lower confidence values.
    """
    n_q = t.shape[1]
    t_mean = t.mean(dim=1)
    t_std = t.std(dim=1)
    t_up = t_mean + ((1.96 * t_std) / np.sqrt(n_q))
    t_low = t_mean - ((1.96 * t_std) / np.sqrt(n_q))

    return to_numpy(t_mean), to_numpy(t_up), to_numpy(t_low)