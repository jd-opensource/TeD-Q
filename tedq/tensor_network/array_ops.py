import numpy as np
import itertools


def get_diag_axes(x, atol=1e-12):
    """Try and find a pair of axes of ``x`` in which it is diagonal.
    Parameters
    ----------
    x : array-like
        The array to search.
    atol : float, optional
        Tolerance with which to compare to zero.
    Returns
    -------
    tuple[int] or None
        The two axes if found else None.
    Examples
    --------
        >>> x = np.array([[[1, 0], [0, 2]],
        ...               [[3, 0], [0, 4]]])
        >>> get_diag_axes(x)
        (1, 2)
    Which means we can reduce ``x`` without loss of information to:
        >>> np.einsum('abb->ab', x)
        array([[1, 2],
               [3, 4]])
    """
    shape = x.shape
    if len(shape) < 2:
        return None

    indxrs = np.indices(shape)

    for i, j in itertools.combinations(range(len(shape)), 2):
        if shape[i] != shape[j]:
            continue
        if np.allclose(x[indxrs[i] != indxrs[j]], 0.0, atol=atol):
            return (i, j)
    return None



def get_antidiag_axes(x, atol=1e-12):
    """Try and find a pair of axes of ``x`` in which it is diagonal.
    Parameters
    ----------
    x : array-like
        The array to search.
    atol : float, optional
        Tolerance with which to compare to zero.
    Returns
    -------
    tuple[int] or None
        The two axes if found else None.
    Examples
    --------
        >>> x = np.array([[[1, 0], [0, 2]],
        ...               [[3, 0], [0, 4]]])
        >>> get_diag_axes(x)
        (1, 2)
    Which means we can reduce ``x`` without loss of information to:
        >>> np.einsum('abb->ab', x)
        array([[1, 2],
               [3, 4]])
    """
    shape = x.shape
    if len(shape) < 2:
        return None

    indxrs = np.indices(shape)

    for i, j in itertools.combinations(range(len(shape)), 2):
        if shape[i] != shape[j]:
            continue
        if np.allclose(x[indxrs[i] != shape[j] - 1 - indxrs[j]], 0.0, atol=atol):
            return (i, j)
    return None