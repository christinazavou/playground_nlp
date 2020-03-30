# -*- coding: utf-8 -*-
import logging
import os
import re
import time
import warnings
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, Counter
from multiprocessing import Pool, Queue
from queue import Full

import numpy as np
from scipy.sparse.csr import csr_matrix

from src.clustering.evaluation import ClusterEvaluator
from src.preprocessing.StemToWord import StemToWord
from src.utils.generic_utils import chunk_serial, get_random_sample
from src.utils.io_utils import read_pickle, write_pickle, write_json, read_json, read_df
from src.utils.logger_utils import get_logger
from src.utils.pandas_utils import iter_tickets_on_field, chunk_dataframe_serial, append_column
from src.visualization.high_dim_data_projection import run_tsne_projection
from src.visualization.topic import topic_cloud

warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

LOGGER = get_logger(__name__, logging.INFO)


def worker_fit_labels(input_queue, result_queue):
    """
    Perform get_fit_labels for each (chunk_no, chunk, model) 3-tuple from the
    input queue, placing the resulting state into the result queue.
    """
    LOGGER.debug("Worker process entering worker_fit_labels loop")
    while True:
        LOGGER.debug("Getting a new job")
        chunk_no, chunk, worker_model, chunk_first, chunk_last = input_queue.get()

        zero_data = np.zeros(len(chunk))
        labels, distributions, probabilities = zero_data, zero_data.astype(object), zero_data.astype(tuple)
        LOGGER.debug("Processing chunk #%i of %i documents", chunk_no, len(chunk))

        for it, i in enumerate(chunk):
            labels[it], distributions[it], probabilities[it] = worker_model.predict(i, idx=it)

        del chunk
        LOGGER.debug("Processed chunk, queuing the result")
        result_queue.put((labels, distributions, probabilities, chunk_first, chunk_last))
        del worker_model
        LOGGER.debug("Result put")


def worker_find_cluster_and_text(input_queue, result_queue):
    # todo: add use of degree_field
    LOGGER.debug("Worker process entering worker_fit_labels loop")
    while True:
        LOGGER.debug("Getting a new job")
        chunk_no, chunk_df, cluster_field, text_field, degree_field, use_bi_grams = input_queue.get()
        LOGGER.debug("processing chunk #%i of %i documents", chunk_no, len(chunk_df))
        cluster_indices, cluster_texts_with_degree, cluster_texts = {}, {}, {}
        for cluster, cluster_df in chunk_df.groupby(cluster_field):
            cluster = int(cluster)
            cluster_indices.setdefault(cluster, [])

            cluster_indices[cluster] += [
                u'idx {} instance {}'.format(idx, row[u'instanceId']) for idx, row in cluster_df.iterrows()
            ]

            cluster_texts.setdefault(cluster, u'')
            cluster_texts[cluster] += \
                u' '.join([text for _, text in iter_tickets_on_field(text_field, cluster_df, use_bi_grams, False)]) \
                + u' '

        del chunk_df
        LOGGER.debug("Processed chunk, queuing the result")
        result_queue.put((cluster_indices, cluster_texts, cluster_texts_with_degree))
        LOGGER.debug("Result put")


class ClusterParameters:

    def __init__(self, num_clusters=30, use_bi_grams=False, max_df=0.95, min_df=5, max_features=None):
        self.num_clusters = num_clusters
        self.use_bi_grams = use_bi_grams
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features


class ClusterModel:

    __metaclass__ = ABCMeta

    def __init__(self, datafile, field, model_dir, stw_file, model_name, cluster_parameters=None):
        LOGGER.info('Building/loading {} model...'.format(model_name))

        self.data_file = datafile
        self.field = field

        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        self.model_dir = model_dir
        self.stw = StemToWord(stw_file=stw_file)
        self.model_name = model_name

        self.model_file = os.path.join(self.model_dir, self.model_name + ".pkl.gzip")

        self.model = None
        self.corpus = None

        self.cluster_parameters = cluster_parameters if cluster_parameters else ClusterParameters()

        self.scores = {}  # dictionary that keeps the evaluation of the cluster

        # comment this if you use python < 3.7
        self.logger = get_logger(__name__, 'INFO', self.model_dir, self.model_name + ".log")

    def get_logger(self):
        """
        NOTE: since python 3.7 loggers can be pickled. before they couldn't
        """
        # return get_logger(__name__, 'INFO', os.path.dirname(os.path.realpath(self.model_file)), 'train.log')
        # comment this if you use python < 3.7 and use the line above
        return self.logger

    @abstractmethod
    def assign_to_cluster(self, doc, **kwargs):
        pass

    @abstractmethod
    def get_vocabulary_tokens(self, split=True, stw=None, use_stw=False):
        pass

    @abstractmethod
    def get_corpus(self, sparse=True):
        pass

    @abstractmethod
    def get_sorted_clusters_labels(self, **kwargs):
        pass

    def load(self):
        tmp_dict = read_pickle(self.model_file)
        self.__dict__.update(tmp_dict)
        LOGGER.info('Length of corpus {}'.format(self.corpus_length()))

    def corpus_length(self):
        if isinstance(self.corpus, csr_matrix):
            return self.corpus.shape[0]
        else:
            return len(self.corpus)

    def save(self):
        """
        NOTE: since python 3.7 loggers can be pickled. before they couldn't
        """
        write_pickle(self.__dict__, self.model_file)

    def store_clusters(self, clusters_file, clusters_text_file, cluster_field=None, degree_field=None, workers=2):
        if os.path.isfile(clusters_file) and os.path.isfile(clusters_text_file):
            return  # do not overwrite
        cluster_field = cluster_field if cluster_field else 'cluster{}'.format(self.model_name)
        cluster_indices, cluster_texts, cluster_texts_with_degree = self.create_clusters_parallel(
            self.data_file, self.field, cluster_field, self.cluster_parameters.use_bi_grams, degree_field=degree_field,
            workers=workers, chunksize=1000
        )
        write_json(cluster_indices, clusters_file)
        write_json(cluster_texts, clusters_text_file)
        # if cluster_texts_with_degree:
        #     store_json(cluster_texts_with_degree, clusters_text_file.replace('.json', '.withDegree.json'))
        # todo: use cluster_texts_with_degree

    def evaluate(self, metric='silhouette', **kwargs):  # currently only silhouette_coefficient evaluation!
        if metric in self.scores:
            return self.scores[metric]

        if metric == 'silhouette':
            labels = np.array(self.fit_labels_parallel(**kwargs)[0])
            corpus = self.get_corpus(True)

            samples = min(2000, int(corpus.shape[0] * 0.5))
            corpus, labels = get_random_sample(corpus, labels, samples)

            self.scores[metric] = ClusterEvaluator().silhouette_coefficient(corpus, labels, self.get_logger())

        elif metric == 'cohesion':
            # todo: add assertion that model is LDA
            words_ids_per_class = self.words_ids_per_topic()
            self.scores[metric] = ClusterEvaluator().topics_cohesion_sparse(self.get_sparse_corpus(),
                                                                            self.get_vocabulary_ids(),
                                                                            words_ids_per_class,
                                                                            self.get_logger())
        self.save()
        return self.scores[metric]

    def visualize_topic_clouds(self, folder, min_cohesion):
        LOGGER.info('Visualizing topic clouds ...')
        clusters_labels = self.get_sorted_clusters_labels(use_stw=True, with_degree=True, num_words=10)
        if 'cohesion' not in self.scores:
            return
        for cluster, labels in clusters_labels.items():
            if self.scores['cohesion'][cluster] < min_cohesion:
                continue
            topic_cloud_file = os.path.join(folder, 'cluster_{}.png'.format(cluster))
            if os.path.isfile(topic_cloud_file):
                continue  # do not overwrite
            if labels:
                topic_cloud(labels, topic_cloud_file)

    def visualize_t_sne(self, **kwargs):

        if not os.path.isfile(self.model_file.replace('.pkl', 't-sne.png')):
            labels, _, _ = self.get_labels_as_list(**kwargs)
            if not labels:
                labels, _, _ = self.fit_labels_parallel()

            for verbose, early_ex, n_iter, metric, init, random_st in zip([1, 1, 1],
                                                                          [4, 6, 8],
                                                                          [500, 1000, 5000],
                                                                          ['cosine', 'cosine', 'cosine'],
                                                                          ['pca', 'random', 'pca'],
                                                                          [200, 200, 200]):
                run_tsne_projection(self.get_corpus(True),
                                    self.model_file.replace('.pkl', 't-sne.png'),
                                    self.get_logger(),
                                    np.array(labels),
                                    verbose=verbose,
                                    early_exaggeration=early_ex,
                                    n_iter=n_iter,
                                    metric=metric,
                                    init=init,
                                    random_state=random_st)

    def get_labels_as_list(self, **kwargs):
        """
        :param kwargs:
        :return: three lists: labels, distributions, degrees
        """
        df = read_df(self.data_file)
        existing_columns = df.columns

        labels, distributions, degrees = None, None, None

        cluster_column = 'cluster{}'.format(self.model_name)
        degree_column = 'degree{}'.format(self.model_name)
        distribution_column = 'distribution{}'.format(self.model_name)

        to_show_degrees = 'show_degree' in kwargs.keys() and kwargs['show_degree']
        to_show_distributions = 'show_distribution' in kwargs.keys() and kwargs['show_distribution']

        if cluster_column in existing_columns:
            labels = df[cluster_column].tolist()
        if degree_column in existing_columns and to_show_degrees:
            degrees = df[degree_column].tolist()
        if distribution_column in existing_columns and to_show_distributions:
            distributions = df[distribution_column].tolist()

        return labels, distributions, degrees

    def save_fitted_labels_as_list_parallel(self, workers=2, chunksize=10000, **kwargs):
        """
        :param workers:
        :param chunksize:
        :param kwargs:
        :return:
        """
        LOGGER.root.level = logging.DEBUG

        labels, distributions, degrees = self.get_labels_as_list(**kwargs)
        if labels:
            return labels, distributions, degrees

        s_time = time.time()
        grouper = chunk_serial

        job_queue = Queue(2 * workers)
        result_queue = Queue()

        pool = Pool(workers, worker_fit_labels, (job_queue, result_queue,))
        queue_size = [0]  # integer can't be accessed in inner definition so list is used instead

        zeros = np.zeros(self.corpus_length())
        labels, distributions, degrees = zeros, zeros.astype(object), zeros.astype(tuple)

        def process_result_queue():
            """Clear the result queue, merging all intermediate results"""
            while not result_queue.empty():
                _labels, _distributions, _probabilities, _chunk_first_idx, _chunk_last_idx = result_queue.get()
                labels[_chunk_first_idx:_chunk_last_idx] = _labels
                distributions[_chunk_first_idx:_chunk_last_idx] = _distributions
                degrees[_chunk_first_idx:_chunk_last_idx] = _probabilities
                queue_size[0] -= 1

        chunk_stream = grouper(self.corpus, chunksize)
        chunk_first_idx, chunk_last_idx = 0, 0
        for chunk_no, chunk in enumerate(chunk_stream):
            chunk_first_idx = chunk_last_idx
            chunk_last_idx = chunk_first_idx + len(chunk)  # keep track of how many documents we've processed so far

            # put the chunk into the workers' input job queue
            chunk_put = False
            while not chunk_put:
                try:
                    job_queue.put((chunk_no, chunk, self, chunk_first_idx, chunk_last_idx), block=False)
                    chunk_put = True
                    queue_size[0] += 1
                    logging.info('PROGRESS: dispatched chunk #{} = '
                                 'documents {}-{}, outstanding queue size {}'.format(
                        chunk_no, chunk_first_idx, chunk_last_idx, queue_size[0]))
                except Full:
                    # in case the input job queue is full, keep clearing the result queue to make sure we don't deadlock
                    process_result_queue()

            process_result_queue()

        while queue_size[0] > 0:  # wait for all outstanding jobs to finish
            process_result_queue()
        if chunk_last_idx != self.corpus_length():
            raise RuntimeError("Input corpus size changed during fit_labels_parallel")

        pool.terminate()
        logging.info('Finding fitted labels finished after {} minutes'.format((time.time() - s_time) // 60))

        self.save_labels(labels, distributions, degrees, **kwargs)
        logging.info(
            'Parallel assigning of fitted labels finished after {} minutes'.format((time.time() - s_time) // 60))
        return labels, distributions, degrees

    def save_labels(self, labels, distributions, degrees, **kwargs):
        df = read_df(self.data_file)
        existing_columns = df.columns
        to_show_distributions = 'show_distribution' in kwargs.keys() and kwargs['show_distribution']
        if to_show_distributions and 'distribution{}'.format(self.model_name) not in existing_columns:
            append_column(self.data_file, 'distribution{}'.format(self.model_name), distributions)
        to_show_degrees = 'show_degree' in kwargs.keys() and kwargs['show_degree']
        if to_show_degrees and 'degree{}'.format(self.model_name) not in existing_columns:
            append_column(self.data_file, 'degree{}'.format(self.model_name), degrees)
        if 'cluster{}'.format(self.model_name) not in existing_columns:
            append_column(self.data_file, 'cluster{}'.format(self.model_name), labels)

    @staticmethod
    def create_clusters_parallel(input_file, text_field, cluster_field, use_bi_grams, degree_field=None,
                                 workers=2, chunksize=1000):

        grouper = chunk_dataframe_serial

        job_queue = Queue(2 * workers)
        result_queue = Queue()

        pool = Pool(workers, worker_find_cluster_and_text, (job_queue, result_queue,))
        queue_size = [0]  # integer can't be accessed in inner definition so list used
        cluster_indices_total = {}
        cluster_texts_total = {}
        cluster_texts_with_degree_total = {}

        def process_result_queue():
            """Clear the result queue, merging all intermediate results"""
            while not result_queue.empty():
                res = result_queue.get()
                for cluster_name, cluster_value in res[0].iteritems():
                    cluster_indices_total.setdefault(cluster_name, [])
                    cluster_indices_total[cluster_name] += cluster_value
                for cluster_name, cluster_value in res[1].iteritems():
                    cluster_texts_total.setdefault(cluster_name, u'')
                    cluster_texts_total[cluster_name] += cluster_value
                if degree_field:
                    for cluster_name, cluster_value in res[2].iteritems():
                        cluster_texts_with_degree_total.setdefault(cluster_name, u'')
                        cluster_texts_with_degree_total[cluster_name] += cluster_value
                queue_size[0] -= 1

        read_columns = [text_field, cluster_field, u'instanceId']
        read_columns += [degree_field] if degree_field else []
        df = read_df(input_file, read_columns=read_columns)
        chunk_stream = grouper(df, chunksize)
        total_documents = len(df)
        del df

        for chunk_no, chunk in enumerate(chunk_stream):
            # put the chunk into the workers' input job queue
            chunk_put = False
            while not chunk_put:
                try:
                    job_queue.put((chunk_no, chunk, cluster_field, text_field, degree_field, use_bi_grams), block=False)
                    chunk_put = True
                    queue_size[0] += 1
                    LOGGER.info('PROGRESS: dispatched chunk {} = '
                                'documents up to {}, over {} documents, outstanding queue size {}'.
                                format(chunk_no, chunk_no * chunksize + len(chunk), total_documents, queue_size[0]))
                except Full:
                    # in case the input job queue is full, keep clearing the result queue to make sure we don't deadlock
                    process_result_queue()

            process_result_queue()

        while queue_size[0] > 0:  # wait for all outstanding jobs to finish
            process_result_queue()

        pool.terminate()

        for key, value in cluster_indices_total.items():
            cluster_indices_total[key] = {'tickets': value}
        for key, value in cluster_texts_total.items():
            cluster_texts_total[key] = re.sub(r' +', u' ', value)
        if degree_field:
            for key, value in cluster_texts_with_degree_total.items():
                cluster_texts_with_degree_total[key] = re.sub(r' +', u' ', value)
        return cluster_indices_total, cluster_texts_total, cluster_texts_with_degree_total

    @staticmethod
    def get_field_distribution_per_cluster(input_file, text_field, cluster_field, top_n=30):
        read_columns = [text_field, cluster_field, u'instanceId']
        df = read_df(input_file, read_columns=read_columns)
        field_per_cluster = {}
        for cluster, cluster_df in df.groupby(cluster_field):
            field_per_cluster[cluster] = Counter(
                [re.sub(r' +', u' ', text, re.U)
                 for _, text in iter_tickets_on_field(text_field, df=cluster_df, as_list=False)]
            )
            field_per_cluster[cluster] = field_per_cluster[cluster].most_common(top_n)
        return field_per_cluster

    @staticmethod
    def save_counts(clusters_file):
        LOGGER.info('updating dict for counts')
        cluster_dict = read_json(clusters_file)
        if '1' in cluster_dict.keys():
            if isinstance(cluster_dict['1'], dict) and 'count' in cluster_dict['1'].keys():
                return
        for cluster, values in cluster_dict.iteritems():
            if 'tickets' in values.keys():
                cluster_dict[cluster]['count'] = len(values['tickets'])
            else:
                cluster_dict[cluster]['count'] = 0
        write_json(cluster_dict, clusters_file)

    @staticmethod
    def order_by(clusters_file, by='count'):
        LOGGER.info('sorting dict by {}'.format(by))
        cluster_dict = read_json(clusters_file)
        to_sort = [(cluster, float(cluster_dict[cluster][by])) for cluster in cluster_dict.keys()]
        sorted_clusters_values = sorted(to_sort, key=lambda tup: tup[1], reverse=True)
        ordered_cluster_dict = OrderedDict()
        for cluster, _ in sorted_clusters_values:
            ordered_cluster_dict[cluster] = cluster_dict[cluster]
        write_json(ordered_cluster_dict, clusters_file)

# todo: add rest functions..
