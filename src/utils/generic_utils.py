import itertools


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
