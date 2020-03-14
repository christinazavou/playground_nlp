# -*- coding: utf-8 -*-
import re
import os
import time
import logging
import warnings
import numpy as np
from queue import Full
from multiprocessing import Pool, Queue
from abc import ABCMeta, abstractmethod
from collections import OrderedDict, Counter

from src.pre_process.StemToWord import StemToWord
from src.visualization.topic_visualization import topic_cloud
from .evaluation import topics_cohesion_sparse, silhouette_coefficient
from src.utils.utils import read_df, store_df, iter_tickets_on_field, append_column, dict_to_utf, \
    iter_tickets_with_degree, bi_grams_on_sentences_list, eval_utf, chunk_serial, manage_logger, chunk_df_serial, \
    read_json, store_json
from src.utils.store_load_zipped import load_pickle, store_pickle
from src.visualization.high_dim_data_projection import project_wrapper


warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')
logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.INFO)
logging.root.level = logging.INFO


def worker_fit_labels(input_queue, result_queue):
    """
    Perform get_fit_labels for each (chunk_no, chunk, model) 3-tuple from the
    input queue, placing the resulting state into the result queue.
    """
    logging.debug("worker process entering worker_fit_labels loop")
    while True:
        logging.debug("getting a new job")
        chunk_no, chunk, worker_model, chunk_first, chunk_last = input_queue.get()
        # labels, distributions, probabilities = [], [], []
        labels, distributions, probabilities = np.zeros(len(chunk)), np.zeros(len(chunk)).astype(object), np.zeros(len(chunk)).astype(tuple)
        logging.debug("processing chunk #%i of %i documents", chunk_no, len(chunk))
        for it, i in enumerate(chunk):
            labels[it], distributions[it], probabilities[it] = worker_model.predict(i, idx=it)
        del chunk
        logging.debug("processed chunk, queuing the result")
        result_queue.put((labels, distributions, probabilities, chunk_first, chunk_last))
        del worker_model
        logging.debug("result put")


def worker_find_cluster_and_text(input_queue, result_queue):
    logging.debug("worker process entering worker_fit_labels loop")
    while True:
        logging.debug("getting a new job")
        chunk_no, chunk_df, cluster_field, text_field, degree_field, use_bi_grams = input_queue.get()
        logging.debug("processing chunk #%i of %i documents", chunk_no, len(chunk_df))
        cluster_indices, cluster_texts_with_degree, cluster_texts = {}, {}, {}
        for cluster, cluster_df in chunk_df.groupby(cluster_field):
            # if cluster == 498.:
            #     print cluster_df['instanceId'].values, '\n'
            cluster = int(cluster)
            cluster_indices.setdefault(unicode(cluster), [])
            if degree_field:
                cluster_indices[unicode(cluster)] += [
                    u'idx {} instance {} assignment {} {}'.format(
                        idx, row[u'instanceId'], cluster, row[degree_field])
                    for idx, row in cluster_df.iterrows()
                    ]
            else:
                cluster_indices[unicode(cluster)] += [
                    u'idx {} instance {}'.format(idx, row[u'instanceId']) for idx, row in cluster_df.iterrows()
                    ]

            if degree_field:
                cluster_texts_with_degree.setdefault(unicode(cluster), u'')
                cluster_texts_with_degree[unicode(cluster)] += \
                    u' '.join([text for _, text in iter_tickets_with_degree(
                        text_field, degree_field, data=cluster_df, use_bi_grams=use_bi_grams)]
                              ) + u' '

            cluster_texts.setdefault(unicode(cluster), u'')
            cluster_texts[unicode(cluster)] += \
                u' '.join([text for _, text in iter_tickets_on_field(
                    text_field, data=cluster_df, use_bi_grams=use_bi_grams, as_list=False)]
                          ) + u' '

        del chunk_df
        logging.debug("processed chunk, queuing the result")
        result_queue.put((cluster_indices, cluster_texts, cluster_texts_with_degree))
        logging.debug("result put")


class ClusterModel:
    __metaclass__ = ABCMeta

    def __init__(self, datafile, field, model_file, stw_file, name, num_clusters=30, use_bi_grams=False, max_df=0.95,
                 min_df=5, max_features=None):
        logging.info('building/loading {} model...'.format(name))
        self.data_file = datafile
        self.field = field
        self.num_clusters = num_clusters
        self.model_file = model_file
        self.use_bi_grams = use_bi_grams
        self.stw = StemToWord(stw_file=stw_file)
        self.name = name
        self.model = None
        self.corpus = None
        self.max_df = max_df
        self.min_df = min_df
        self.max_features = max_features

    @abstractmethod
    def predict(self, doc, **kwargs):
        pass

    @abstractmethod
    def get_vocabulary_tokens(self, split=True, stw=None, use_stw=False):
        pass

    @abstractmethod
    def get_sorted_clusters_labels(self, **kwargs):
        pass

    # @abstractmethod
    # def get_fitted_labels(self, **kwargs):
    #     pass

    def replace_file(self, with_file):
        os.rename(self.data_file, 'tmp.csv')
        os.rename(with_file, self.data_file)
        os.remove('tmp.csv')

    def load(self):
        tmp_dict = load_pickle(self.model_file)
        self.__dict__.update(tmp_dict)
        try:
            logging.info('length of corpus {}'.format(self.corpus.shape[0]))
        except:
            logging.info('length of corpus {}'.format(len(self.corpus)))

    def save(self):
        store_pickle(self.__dict__, self.model_file, 2)

    def get_dense_corpus(self):
        pass

    def get_sparse_corpus(self):
        pass

    def find_clusters_and_text(self, clusters_file, clusters_text_file, input_file=None, text_field=None,
                               cluster_field=None, use_bi_grams=None, degree_field=None, workers=2):
        if os.path.isfile(clusters_file) and os.path.isfile(clusters_text_file):
            if degree_field and os.path.isfile(clusters_text_file.replace('.json', '.withdegree.json')):
                return
            if not degree_field:
                return
        if not input_file:
            input_file = self.data_file
        if not text_field:
            text_field = self.field
        if not cluster_field:
            cluster_field = 'cluster{}'.format(self.name)
        if not use_bi_grams:
            use_bi_grams = self.use_bi_grams
        cluster_indices, cluster_texts, cluster_texts_with_degree = self.get_clusters_and_texts_parallel(
            input_file, text_field, cluster_field, use_bi_grams, degree_field=degree_field,
            workers=workers, chunksize=1000
        )
        store_json(cluster_indices, clusters_file)
        store_json(cluster_texts, clusters_text_file)
        if cluster_texts_with_degree:
            store_json(cluster_texts_with_degree, clusters_text_file.replace('.json', '.withdegree.json'))

    def save_clusters_labels_json(self, clusters_labels_file, **kwargs):
        logging.info('saving the clusters labels...')
        clusters_dict = {}
        if os.path.isfile(clusters_labels_file):
            clusters_dict = read_json(clusters_labels_file)
        if '1' in clusters_dict.keys():
            if isinstance(clusters_dict['1'], dict) and 'clusters_labels' in clusters_dict['1'].keys():  # cluster 1 should always exists
        # if key_exists(clusters_dict, 'clusters_labels'):
                logging.info('clusters labels exist')
                return
        clusters_labels = self.get_sorted_clusters_labels(**kwargs)
        clusters_dict = append_to_json_clusters('clusters_labels', dict_to_utf(clusters_labels), clusters_dict)
        store_json(clusters_dict, clusters_labels_file)

    # def save_fitted_labels(self, **kwargs):
    #     list_df = list(read_df(self.data_file))
    #     if 'cluster{}'.format(self.name) not in list_df or (
    #         'show_degree' in kwargs.keys() and kwargs['show_degree'] and 'degree{}'.format(self.name) not in list_df
    #     ) or (
    #         'show_distr' in kwargs.keys() and kwargs['show_distr'] and 'distribution{}'.format(self.name) not in list_df
    #     ):
    #         labels, distributions, degrees = self.get_fitted_labels(**kwargs)
    #         if 'show_distr' in kwargs.keys() and kwargs['show_distr'] and 'distribution{}'.format(self.name) not in list_df:
    #             append_column(self.data_file, 'distribution{}'.format(self.name), distributions)
    #         if 'show_degree' in kwargs.keys() and kwargs['show_degree'] and 'degree{}'.format(self.name) not in list_df:
    #             append_column(self.data_file, 'degree{}'.format(self.name), degrees)
    #         if 'cluster{}'.format(self.name) not in list_df:
    #             append_column(self.data_file, 'cluster{}'.format(self.name), labels)
    #     else:
    #         df = read_df(self.data_file)
    #         labels, distributions, degrees = [], [], []
    #         if 'distribution{}'.format(self.name) in list(df):
    #             distributions = df['distribution{}'.format(self.name)].tolist()
    #         if 'degree{}'.format(self.name) in list(df):
    #             degrees = df['degree{}'.format(self.name)].tolist()
    #         labels = df['cluster{}'.format(self.name)].tolist()
    #     return labels, distributions, degrees

    def evaluate(self, metric='silhouette', **kwargs):  # currently only silhouette_coefficient evaluation!
        logger = manage_logger(__name__, 'INFO', os.path.dirname(os.path.realpath(self.model_file)), 'train.log')
        if metric == 'silhouette':
            if 'silhouette' not in self.__dict__:
                s_time = time.time()
                labels = np.array(self.save_fitted_labels_parallel(**kwargs)[0])
                corpus = self.get_sparse_corpus()
                samples = corpus.shape[0]
                samples = int(samples*0.5)
                if samples > 2000:
                    samples = 2000
                labels = labels[0:samples]
                logger.info('evaluating the silhouette coefficient on {} samples...'.format(samples))
                corpus = corpus[0:samples]
                c = silhouette_coefficient(corpus, labels)
                logger.info('calculations finished after {} seconds. '.format(time.time()-s_time))
                self.silhouette = c
                logger.info('silhouette coefficient = {}'.format(self.silhouette))
            self.save()
            logging.info('silhouette coefficient = {}'.format(self.silhouette))
            return self.silhouette
        elif metric == 'cohesion':
            if 'cohesion_per_topic' not in self.__dict__:
                s_time = time.time()
                logger.info('calculating cohesion per topic...')
                words_ids_per_class = self.words_ids_per_topic()
                self.cohesion_per_topic = topics_cohesion_sparse(self.get_sparse_corpus(), words_ids_per_class)
                self.save()
                logger.info('finished calculations after {} seconds.'.format(time.time()-s_time))
                logger.info('cohesions = {}'.format(self.cohesion_per_topic))
            logging.info('cohesions = {}'.format(self.cohesion_per_topic))
            return self.cohesion_per_topic

    def visualize(self, folder, **kwargs):
        logging.info('visualizing clusters ...')
        clusters_labels = self.get_sorted_clusters_labels(use_stw=True, with_degree=True, num_words=10)
        for cluster, labels in clusters_labels.iteritems():
            if 'min_cohesion' in kwargs.keys() and 'cohesion_per_topic' in self.__dict__.keys() and \
                            self.cohesion_per_topic[cluster] < kwargs['min_cohesion']:
                continue
            if not os.path.isfile(os.path.join(folder, 'cluster_{}.png'.format(cluster))) and labels != []:
                try:
                    topic_cloud(labels, os.path.join(folder, 'cluster_{}.png'.format(cluster)))
                except:
                    print 'could not create figure for topic {} with labels {}'.format(cluster, labels)
        if 't_sne' in kwargs.keys() and kwargs['t_sne']:
            if not os.path.isfile(self.model_file.replace('.p', 't-sne.png')):
                logger = manage_logger(__name__, 'INFO', os.path.dirname(os.path.realpath(self.model_file)), 'train.log')

                labels, _, _ = self.labels_exists(**kwargs)
                if not labels:
                    labels, _, _ = self.save_fitted_labels_parallel()

                project_wrapper(self.get_sparse_corpus(), self.model_file.replace('.p', 't-sne.png'), logger, np.array(labels),
                                verbose=[1, 1, 1], early_exaggeration=[4, 6, 8],
                                n_iter=[500, 1000, 5000], metric=['cosine', 'cosine', 'cosine'],
                                init=['pca', 'random', 'pca'], random_state=[200, 200, 200])

    def save_fitted_labels_parallel(self, workers=2, chunksize=10000, **kwargs):

        logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)
        logging.root.level = logging.DEBUG

        labels, distributions, degrees = self.labels_exists(**kwargs)
        if labels:
            return labels, distributions, degrees

        s_time = time.time()
        grouper = chunk_serial  # choose according to a parameter which function to call when call grouper() !!!!

        job_queue = Queue(2*workers)
        result_queue = Queue()

        pool = Pool(workers, worker_fit_labels, (job_queue, result_queue,))
        queue_size, real_len = [0], 0  # integer can't be accessed in inner definition so list used
        # labels, distributions, degrees = [], [], []
        labels, distributions, degrees = \
            np.zeros(len(self.corpus)), np.zeros(len(self.corpus)).astype(object), np.zeros(len(self.corpus)).astype(tuple)

        def process_result_queue():
            """Clear the result queue, merging all intermediate results"""
            while not result_queue.empty():
                lab, distr, degr, ch_f, ch_l = result_queue.get()
                labels[ch_f:ch_l] = lab
                distributions[ch_f:ch_l] = distr
                degrees[ch_f:ch_l] = degr
                queue_size[0] -= 1

        chunk_stream = grouper(self.corpus, chunksize)
        chunk_first, chunk_last = 0, 0
        for chunk_no, chunk in enumerate(chunk_stream):
            chunk_first = chunk_last
            chunk_last = chunk_first + len(chunk)

            real_len += len(chunk)  # keep track of how many documents we've processed so far

            # put the chunk into the workers' input job queue
            chunk_put = False
            while not chunk_put:
                try:
                    # job_queue.put((chunk_no, chunk, self.model, chunk_first, chunk_last), block=False)
                    job_queue.put((chunk_no, chunk, self, chunk_first, chunk_last), block=False)
                    chunk_put = True
                    queue_size[0] += 1
                    logging.info('PROGRESS: dispatched chunk #{} = '
                                 'documents {}-{}, outstanding queue size {}'.format(
                                  chunk_no, chunk_first, chunk_last, queue_size[0]))
                except Full:
                    # in case the input job queue is full, keep clearing the result queue to make sure we don't deadlock
                    process_result_queue()

            process_result_queue()

        while queue_size[0] > 0:  # wait for all outstanding jobs to finish
            process_result_queue()
        if real_len != len(self.corpus):
            raise RuntimeError("input corpus size changed during fit_labels_parallel")

        pool.terminate()
        logging.info('finding fitted labels finished after {} mins'.format((time.time()-s_time)//60))

        list_df = list(read_df(self.data_file))
        if 'show_distr' in kwargs.keys() and kwargs['show_distr'] and 'distribution{}'.format(self.name) not in list_df:
            append_column(self.data_file, 'distribution{}'.format(self.name), distributions)
        if 'show_degree' in kwargs.keys() and kwargs['show_degree'] and 'degree{}'.format(self.name) not in list_df:
            append_column(self.data_file, 'degree{}'.format(self.name), degrees)
        if 'cluster{}'.format(self.name) not in list_df:
            append_column(self.data_file, 'cluster{}'.format(self.name), labels)
        logging.info('assigning fitted labels finished after {} mins'.format((time.time()-s_time)//60))
        return labels, distributions, degrees

    def labels_exists(self, **kwargs):
        list_df = list(read_df(self.data_file))
        if 'cluster{}'.format(self.name) not in list_df or (
            'show_degree' in kwargs.keys() and kwargs['show_degree'] and 'degree{}'.format(self.name) not in list_df
        ) or (
            'show_distr' in kwargs.keys() and kwargs['show_distr'] and 'distribution{}'.format(self.name) not in list_df
        ):
            return None, None, None
        else:
            df = read_df(self.data_file)
            labels, distributions, degrees = [], [], []
            if 'distribution{}'.format(self.name) in list(df):
                distributions = df['distribution{}'.format(self.name)].tolist()
            if 'degree{}'.format(self.name) in list(df):
                degrees = df['degree{}'.format(self.name)].tolist()
            labels = df['cluster{}'.format(self.name)].tolist()
            return labels, distributions, degrees

    @staticmethod
    def get_clusters_and_texts_parallel(input_file, text_field, cluster_field, use_bi_grams, degree_field=None, workers=2,
                                        chunksize=1000):

        grouper = chunk_df_serial  # choose according to a parameter which function to call when call grouper() !!!!

        job_queue = Queue(2*workers)
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

        # for chunk_no, chunk in enumerate(chunk_stream):
        #     for idx, row in chunk.iterrows():
        #         if row[cluster_field] == 498.:
        #             print row

        for chunk_no, chunk in enumerate(chunk_stream):
            # put the chunk into the workers' input job queue
            chunk_put = False
            while not chunk_put:
                try:
                    job_queue.put((chunk_no, chunk, cluster_field, text_field, degree_field, use_bi_grams), block=False)
                    chunk_put = True
                    queue_size[0] += 1
                    logging.info('PROGRESS: dispatched chunk {} = '
                                 'documents up to {}, over {} documents, outstanding queue size {}'.
                                 format(chunk_no, chunk_no * chunksize + len(chunk), total_documents, queue_size[0]))
                except Full:
                    # in case the input job queue is full, keep clearing the result queue to make sure we don't deadlock
                    process_result_queue()

            process_result_queue()

        while queue_size[0] > 0:  # wait for all outstanding jobs to finish
            process_result_queue()

        pool.terminate()
        # print 'all\n', cluster_indices_total[unicode(498.0)]

        for key, value in cluster_indices_total.iteritems():
            cluster_indices_total[key] = {'tickets': value}
        for key, value in cluster_texts_total.iteritems():
            cluster_texts_total[key] = re.sub(r' +', u' ', value)
        if degree_field:
            for key, value in cluster_texts_with_degree_total.iteritems():
                cluster_texts_with_degree_total[key] = re.sub(r' +', u' ', value)
        return cluster_indices_total, cluster_texts_total, cluster_texts_with_degree_total

    @staticmethod
    def get_field_distribution_per_cluster(input_file, text_field, cluster_field, top_n=30):
        read_columns = [text_field, cluster_field, u'instanceId']
        df = read_df(input_file, read_columns=read_columns)
        field_per_cluster = {}
        for cluster, cluster_df in df.groupby(cluster_field):
            field_per_cluster[unicode(cluster)] = Counter(
                [re.sub(r' +', u' ', text, re.U)
                 for _, text in iter_tickets_on_field(text_field, data=cluster_df, as_list=False)]
            )
            field_per_cluster[unicode(cluster)] = field_per_cluster[unicode(cluster)].most_common(top_n)
        return field_per_cluster

    @staticmethod
    def save_clusters_keys_json(clusters_keys_file, kea_obj, only_predictive=False, only_freq=False, stw=None,
                                init_class=0, key_name='clusters_keys'):
        logging.info('saving the {} ...'.format(key_name))
        clusters_dict = {}
        if os.path.isfile(clusters_keys_file):
            clusters_dict = read_json(clusters_keys_file)
        if '1' in clusters_dict.keys():
            if isinstance(clusters_dict['1'], dict) and key_name in clusters_dict['1'].keys():  # cluster 1 should always exists
                # if key_exists(clusters_dict, key_name):
                logging.info('{} exists'.format(key_name))
                return
        clusters_keys = get_clusters_keys(kea_obj, stw, only_predictive, only_freq, init_class)
        clusters_dict = append_to_json_clusters(key_name, dict_to_utf(clusters_keys), clusters_dict)
        store_json(clusters_dict, clusters_keys_file)

    @staticmethod
    def save_counts(clusters_file):
        logging.info('updating dict for counts')
        cluster_dict = read_json(clusters_file)
        if '1' in cluster_dict.keys():
            if isinstance(cluster_dict['1'], dict) and 'count' in cluster_dict['1'].keys():
                return
        # if key_exists(cluster_dict, 'count'):
        #     return
        for cluster, values in cluster_dict.iteritems():
            if 'tickets' in values.keys():
                cluster_dict[cluster]['count'] = len(values['tickets'])
            else:
                cluster_dict[cluster]['count'] = 0
        store_json(cluster_dict, clusters_file)

    @staticmethod
    def order_by(clusters_file, by='count'):
        logging.info('sorting dict by {}'.format(by))
        cluster_dict = read_json(clusters_file)
        to_sort = [(cluster, float(cluster_dict[cluster][by])) for cluster in cluster_dict.keys()]
        sorted_clusters_values = sorted(to_sort, key=lambda tup: tup[1], reverse=True)
        ordered_cluster_dict = OrderedDict()
        for cluster, _ in sorted_clusters_values:
            ordered_cluster_dict[cluster] = cluster_dict[cluster]
        store_json(ordered_cluster_dict, clusters_file)


def append_to_json_clusters(new_key, clusters_new_values, clusters_dict=None, clusters_dict_file=None):
    if not clusters_dict and clusters_dict_file:
        clusters_dict = read_json(clusters_dict_file)
    if '1' in clusters_dict.keys():
        if isinstance(clusters_dict['1'], dict) and new_key in clusters_dict['1'].keys():  # cluster 1 should always exists
    # if key_exists(clusters_dict, new_key):
            return
    for cluster, new_values in clusters_new_values.iteritems():
        if cluster in clusters_dict.keys():
            tmp = clusters_dict[cluster]
            tmp.update({new_key: new_values})
            clusters_dict[cluster] = tmp
            # clusters_dict[cluster] = cluster_update(clusters_dict[cluster], {new_key: new_values})
        else:
            clusters_dict[cluster] = {new_key: new_values}
    if clusters_dict_file:
        store_json(clusters_dict, clusters_dict_file)
    return clusters_dict


def key_exists(clusters_dict, key):
    if '1' in clusters_dict.keys():
        if isinstance(clusters_dict['1'], dict) and key in clusters_dict['1'].keys():  # cluster 1 should always exists
            return True
    return False


def cluster_update(cluster_dict, new_dict):
    tmp = cluster_dict
    tmp.update(new_dict)
    return tmp


def get_clusters_keys(kea_obj, stw, only_predictive, only_freq, init_class):
    clusters_keys = {}
    for cluster in range(kea_obj.tf.shape[0]):
        if cluster % 50 == 0:
            logging.info('getting keys of cluster {}'.format(cluster))
        keys = kea_obj.get_doc_keywords(
            cluster, 7, stw=stw, only_predictive=only_predictive, only_freq=only_freq, with_degree=True
        )
        sorted_keys = sorted(keys, key=lambda tup: tup[1], reverse=True)
        clusters_keys[cluster + init_class] = sorted_keys
    return clusters_keys


def save_clusters_labels_and_tokenized_text_for_mongo(model, run_id):
    df_list = list(read_df(model.data_file))
    if u'cluster{}-keywords'.format(model.name) in df_list and u'cluster{}-upto20'.format(model.name) in df_list \
            and u'{}-tokenized'.format(model.field) in df_list:
        return
    clusters_keys_up_to_20 = model.get_sorted_clusters_labels(use_stw=False, num_words=20, with_degree=True)
    clusters_keys_to_show = {}
    for cluster in clusters_keys_up_to_20.keys():
        keys_to_show = sorted(clusters_keys_up_to_20[cluster], key=lambda x: x[1], reverse=True)[:5]
        keys_up_to_20 = [key for key, degree in clusters_keys_up_to_20[cluster]]
        keys_to_show = [key for key, degree in keys_to_show]
        keys_to_show = keep_some_grams(keys_to_show)
        for i, key in enumerate(keys_to_show):
            keys = key.split('_')
            keys_to_show[i] = u' '.join([model.stw.find_word(w) for w in keys])
        clusters_keys_up_to_20[cluster] = keys_up_to_20
        clusters_keys_to_show[cluster] = keys_to_show
    df = read_df(model.data_file)
    for idx, row in df.iterrows():
        cluster = int(row[u'cluster{}'.format(model.name)])
        if cluster == -1:
            df.set_value(idx, u'cluster{}-keywords'.format(model.name), unicode('no-keywords'))
            df.set_value(idx, u'cluster{}-upto20'.format(model.name), unicode(''))
            df.set_value(idx, u'{}-tokenized'.format(model.field), unicode(u', '.join(i for i in bi_grams_on_sentences_list(eval_utf(row[model.field])))))
            df.set_value(idx, u'runId', run_id)
            df.set_value(idx, u'cluster{}-clustering_text'.format(model.name), u' '.join(eval_utf(row[model.field.replace('preprocessed', '')])))
        else:
            df.set_value(idx, u'cluster{}-keywords'.format(model.name), unicode(u', '.join(i for i in clusters_keys_to_show[cluster])))
            df.set_value(idx, u'cluster{}-upto20'.format(model.name), unicode(u', '.join(i for i in clusters_keys_up_to_20[cluster])))
            df.set_value(idx, u'{}-tokenized'.format(model.field), unicode(u', '.join(i for i in bi_grams_on_sentences_list(eval_utf(row[model.field])))))
            df.set_value(idx, u'runId', run_id)
            df.set_value(idx, u'cluster{}-clustering_text'.format(model.name), u' '.join(eval_utf(row[model.field.replace('preprocessed', '')])))
    store_df(df, model.data_file)


def real_text(row, field):
    return u' '.join(eval_utf(row[field]))


def keep_some_grams(tokens):
    keep_tokens = tokens
    to_del = []
    for token in keep_tokens:
        if '_' not in token:
            for token2 in keep_tokens:
                if '_' in token2 and token in token2.split('_'):
                    idx = keep_tokens.index(token)
                    if idx not in to_del:
                        to_del.append(idx)
    return_tokens = []
    for idx in range(len(keep_tokens)):
        if idx not in to_del:
            return_tokens.append(keep_tokens[idx])
    return return_tokens
