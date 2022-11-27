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
    x : numpy array, the array to search.
    atol : Tolerance

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


def get_columns(x, atol=1e-12):
    """Find out a pair of index and column that only that column is non-zero
    Parameters
    ----------
    x : numpy array, the array to search.
    atol : float, tolerance
    -------
    tuple[int] or None
        If found, the first integer is which axis, and the second is which
        column of that axis, else None.
    Examples
    --------
        >>> x = np.array([[[0, 1], [0, 2]],
        ...               [[0, 3], [0, 4]]])
        >>> find_columns(x)
        (2, 1)
    Which means we can happily slice ``x`` without loss of information to:
        >>> x[:, :, 1]
        array([[1, 2],
               [3, 4]])
    """
    shape = x.shape
    # scaler
    if len(shape) < 1:
        return None

    indxrs = np.indices(shape)

    for i in range(len(shape)):
        for j in range(shape[i]):
            if np.allclose(x[indxrs[i] != j], 0.0, atol=atol):
                return (i, j)

    return None