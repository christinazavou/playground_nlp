import time

from sklearn.manifold import TSNE

from src.utils.generic_utils import get_random_sample, get_dense_data
from src.utils.logger_utils import log_info
from src.visualization.utils import make_scatter


# SET high perplexity in case of many data where LDA had also high perplexity


def tsne_projection(data, filename, logger=None, y=None, **kwargs):
    """
    :param data:
    :param filename:
    :param logger:
    :param y:
    :param kwargs:
    :return:
    """

    """given non sparse data matrix and file (optional logger and y as numpy array"""

    verbose = kwargs['verbose'] if kwargs and 'verbose' in kwargs else True
    early_exaggeration = kwargs['early_exaggeration'] if kwargs and 'early_exaggeration' in kwargs else 4.0
    n_iter = kwargs['n_iter'] if kwargs and 'n_iter' in kwargs else 1000
    metric = kwargs['metric'] if kwargs and 'metric' in kwargs else "euclidean"
    init = kwargs['init'] if kwargs and 'init' in kwargs else "random"
    random_state = kwargs['random_state'] if kwargs and 'random_state' in kwargs else None

    if verbose:
        log_info('Calculating t-sne...', logger)

    s_time = time.time()
    projection = TSNE(verbose=verbose,
                      early_exaggeration=early_exaggeration,
                      n_iter=n_iter,
                      metric=metric,
                      init=init,
                      random_state=random_state
                      ).fit_transform(data)

    make_scatter(projection, filename, y)

    if verbose:
        log_info('T-sne on data with shape {} finished after {}'.format(data.shape, time.time() - s_time))


def run_tsne_projection(data, filename, logger, y, **kwargs):
    data = get_dense_data(data)
    if data.shape[0] > 5000:
        data, y = get_random_sample(data, y, sample_size=5000)
    y = list(y) if y is not None else None

    tsne_projection(data, filename, logger, y, **kwargs)
