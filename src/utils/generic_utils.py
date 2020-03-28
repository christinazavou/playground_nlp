import itertools

import numpy as np


def chunk_serial(iterable, chunk_size):
    """
    Return elements from the iterable in `chunksize`-ed lists. The last returned
    element may be smaller (if length of collection is not divisible by `chunksize`)
    """
    it = iter(iterable)
    while True:
        wrapped_chunk = [list(itertools.islice(it, int(chunk_size)))]
        if not wrapped_chunk[0]:
            break
        yield wrapped_chunk.pop()


def get_random_sample(x, y=None, sample_size=5000):
    """
    :param x: shape [N, ?, ?, ...]
    :param y: shape [N, ?, ?, ...]
    :param sample_size: int
    :return: shape [sample_size, ?, ?, ...], shape [sample_size, ?, ?, ...]
    """
    sample_indices = np.random.randint(x.shape[0], size=sample_size)
    x = x[sample_indices]
    if y is not None:
        y = y[sample_indices]
    return x, y


def get_dense_data(data):
    if isinstance(data, list):
        return np.array(data)
    if type(data) != np.ndarray:
        return data.toarray()
    return data
