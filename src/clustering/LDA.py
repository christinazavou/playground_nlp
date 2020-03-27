import os
import time
import gensim
import logging
import itertools
from src.utils.utils import read_df
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_selection import VarianceThreshold
from .ClusteringAlgorithms import ClusterModel
from src.utils.utils import iter_tickets_on_field, un_bi_gram, manage_logger


logging.basicConfig(format='%(levelname)s : %(message)s', level=logging.DEBUG)


class LSICorpus(object):
    def __init__(self, corpus, num_topics, id2word):
        # self.model = gensim.models.lsimodel.LsiModel(corpus, num_topics=num_topics, id2word=id2word)
        # for topic in self.model.
        # self.model = gensim.models.lsimodel.stochastic_svd(corpus,)
        pass


class VectorizedCorpus(object):
    def __init__(self, data_file, dictionary, field, use_bi_grams=False):
        """
        Yield each document in turn, as a list of tokens (unicode strings).
        """
        self.data_file = data_file
        self.dictionary = dictionary
        self.field = field
        self.use_bi_grams = use_bi_grams
        self.indices = []

    def __iter__(self):
        for idx, tokens in iter_tickets_on_field(self.field, datafile=self.data_file, use_bi_grams=self.use_bi_grams, as_list=True):
            self.indices.append(idx)
            yield self.dictionary.doc2bow(tokens)

    def __len__(self):
        df = read_df(self.data_file)
        return df.shape[0]


class LDAModel(ClusterModel):
    def __init__(self, data_file, field, model_file, stw_file, name, num_clusters=20, use_bi_grams=False,
                 max_df=0.5, min_df=5, max_features=100000, passes=1, iterations=50, lsi=None, above_variance=None):

        # manage_logger('gensim.corpora.dictionary', 'INFO', os.path.dirname(os.path.realpath(model_file)), 'train.log')
        # manage_logger('gensim.models.ldamodel', 'INFO', os.path.dirname(os.path.realpath(model_file)), 'train.log')
        # manage_logger('gensim.models.ldamodel', 'DEBUG', os.path.dirname(os.path.realpath(model_file)), 'train.log')
        # logger = manage_logger(__name__, 'INFO', os.path.dirname(os.path.realpath(model_file)), 'train.log')
        logger = manage_logger(__name__, 'INFO', '/home/christina/Documents/playground_nlp/', 'templog.log')

        max_df = 0.5 if not max_df else max_df
        min_df = 5 if not min_df else min_df
        max_features = 100000 if not max_features else max_features
        self.iterations = iterations
        self.passes = passes

        super(LDAModel, self).__init__(data_file, field, model_file, stw_file, name, num_clusters, use_bi_grams,
                                       max_df, min_df, max_features)
        self.corpus_file = model_file.replace('_mod_', '_corpus_')

        if not os.path.isfile(model_file):

            doc_stream = (tokens for _, tokens in iter_tickets_on_field(field, datafile=data_file, use_bi_grams=use_bi_grams))

            id2word_data = gensim.corpora.Dictionary(doc_stream)
            id2word_data.filter_extremes(no_below=min_df, no_above=max_df, keep_n=max_features)

            data_corpus = VectorizedCorpus(data_file, id2word_data, field, self.use_bi_grams)  # stream of BOW vectors

            if above_variance:
                init_feature_size = len(id2word_data.keys())
                sparse_corpus = gensim.matutils.corpus2csc(
                    corpus=data_corpus, num_terms=init_feature_size, num_docs=len(data_corpus)
                ).transpose()
                selector = VarianceThreshold(threshold=above_variance)
                selector.fit_transform(sparse_corpus)
                support = selector.get_support()
                keep_ids = [i for i in xrange(init_feature_size) if support[i]]
                id2word_data.filter_tokens(good_ids=keep_ids)
                data_corpus = VectorizedCorpus(data_file, id2word_data, field, self.use_bi_grams)  # stream-BOW vectors
                logger.info('removing features with variance threshold {} leaves us from {} to {} features'.
                            format(above_variance, init_feature_size, len(id2word_data.keys())))

            # sparse_corpus = gensim.matutils.corpus2csc(corpus=data_corpus, num_terms=len(id2word_data.keys()), num_docs=len(data_corpus))
            # svd = TruncatedSVD(n_components=1024)
            # svd.fit(sparse_corpus)
            # a = svd.transform(sparse_corpus)
            # if lsi:
                # self.lsi = LSICorpus(data_corpus, lsi, id2word_data)
                # data_corpus = [self.lsi.model[x] for x in data_corpus]
            # else:
            #     self.lsi = None

            gensim.corpora.MmCorpus.serialize(self.corpus_file, data_corpus)
            self.corpus = gensim.corpora.MmCorpus(self.corpus_file)

            train_size = int(len(self.corpus) * 0.8)
            if train_size < 1000:  # can be more if distributed !?
                train_size = 1000
                passes = 10
            else:
                passes = passes

            clipped_corpus = gensim.utils.ClippedCorpus(self.corpus, train_size)  # fewer documents for training
            s_time = time.time()

            self.model = gensim.models.ldamulticore.LdaMulticore(
                clipped_corpus, num_topics=num_clusters, id2word=id2word_data, workers=2, random_state=40,
                passes=passes, iterations=iterations, eval_every=10
                # , gamma_threshold=0.01  # convergence threshold of gamma to stop
            )

            logger.info(
                'LDA gensim model on {} data finished after {} mins.'.format(train_size, (time.time() - s_time) // 60))
            logger.info('num of features: {}'.format(len(self.model.id2word.keys())))
            self.save()
            # self.update_model()
        else:
            self.load()

    # def perplexity_measure(self):
    #     import numpy as np
    #     indices = np.random.randint(len(self.corpus), size=100)
    #     chunk = [doc for idx, doc in enumerate(self.corpus) if idx in indices]
    #     print self.model.log_perplexity(chunk)

    # def cohesion_measure(self):  # too slow
    #     s_time = time.time()
    #     top_topics = self.model.top_topics(self.corpus, num_words=20)
    #     avg_topic_coherence = sum([t[1] for t in top_topics]) / self.num_clusters
    #     print 'Average gensim topic coherence: {} (took {} mins)'.format(avg_topic_coherence, (time.time()-s_time)//60)
    #     print top_topics

    def update_model(self):
        train_size = int(len(self.corpus) * 0.8)
        if train_size > 100000:
            train_size = 100000
        clipped_corpus = gensim.utils.ClippedCorpus(self.corpus, train_size)  # fewer documents for training
        logger = manage_logger(__name__, 'INFO', os.path.dirname(os.path.realpath(self.model_file)), 'train.log')
        s_time = time.time()
        logger.info('... updating LDA ...')
        self.model.update(clipped_corpus)  # default 50 iterations
        logger.info('update finished after {} mins'.format((time.time() - s_time) // 60))
        self.save()

    # def predict(self, docs, **kwargs):
    #     """input is an iterator of text to be labeled"""
    #     labels = []
    #     for i, x in docs:
    #         vec = self.model.id2word.doc2bow(x)
    #         y = self.model[vec]
    #         if not y:
    #             y = [(-1, 1.0)]
    #         if 'show_distr' in kwargs.keys() and kwargs['show_distr']:
    #             labels.append(y)
    #         else:
    #             sorted_y = sorted(y, key=lambda tup: tup[1], reverse=True)
    #             best_y = sorted_y[0]
    #             if 'show_degree' in kwargs.keys() and kwargs['show_degree']:
    #                 labels.append((best_y[0], round(best_y[1], 4)))
    #             else:
    #                 labels.append(best_y[0])
    #     return labels

    def predict(self, doc, **kwargs):
        y = self.model[doc]
        if not y:
            y = [(-1, 1.0)]
        distribution = y
        sorted_y = sorted(y, key=lambda tup: tup[1], reverse=True)
        best_y = sorted_y[0]
        degree = round(best_y[1], 4)
        label = best_y[0]
        return label, distribution, degree

    def get_sparse_corpus(self):
        return gensim.matutils.corpus2csc(
            self.corpus, num_terms=len(self.model.id2word.keys()), num_docs=len(self.corpus)
        ).transpose()

    def get_dense_corpus(self):
        return gensim.matutils.corpus2dense(self.corpus, num_terms=len(self.model.id2word.keys()), num_docs=len(self.corpus))

    def words_ids_per_topic(self):
        word_ids = {}
        for topic in xrange(self.num_clusters):
            tokens = self.model.get_topic_terms(topic)
            sorted_tokens = sorted(tokens, key=lambda tup: tup[1], reverse=True)
            word_ids[topic] = [token_id for token_id, prob in sorted_tokens[0:10]]
        return word_ids

    def get_fitted_labels(self, **kwargs):
        labels, distributions, probabilities = [], [], []
        for i in range(len(self.corpus)):
            if i % 1000 == 0:
                logging.info('{}-th label'.format(i))
            y = self.model[self.corpus[i]]
            if not y:
                y = [(-1, 1.0)]
            if 'show_distr' in kwargs.keys() and kwargs['show_distr']:
                distributions.append(y)
            sorted_y = sorted(y, key=lambda tup: tup[1], reverse=True)
            best_y = sorted_y[0]
            if 'show_degree' in kwargs.keys() and kwargs['show_degree']:
                probabilities.append((round(best_y[1], 4)))
            labels.append(best_y[0])
        return labels, distributions, probabilities

    def get_vocabulary_tokens(self, split=True, stw=None, use_stw=False):
        if split:
            tokens = [un_bi_gram(token) for token in self.model.id2word.values()]
        else:
            tokens = self.model.id2word.values()
        if use_stw:
            if not stw:
                stw = self.stw
            return [stw.find_word(token) for token in tokens]
        return tokens

    def get_sorted_clusters_labels(self, **kwargs):
        sorted_cluster_labels = {}
        for topic in range(self.num_clusters):
            tokens = self.model.get_topic_terms(topic)
            sorted_tokens = sorted(tokens, key=lambda tup: tup[1], reverse=True)
            sorted_tokens = [(token, prob) for (token, prob) in sorted_tokens if prob != 0]
            sorted_tokens = [(self.model.id2word[token_id], prob) for token_id, prob in sorted_tokens]
            if self.use_bi_grams and 'split' in kwargs and kwargs['split']:
                sorted_tokens = [(un_bi_gram(token), prob) for token, prob in sorted_tokens]
            if 'use_stw' in kwargs.keys() and kwargs['use_stw']:
                sorted_tokens = [(self.stw.find_word(token), prob) for token, prob in sorted_tokens]
            if 'with_degree' in kwargs.keys() and not kwargs['with_degree']:
                sorted_tokens = [token for token, prob in sorted_tokens]
            if 'num_words' in kwargs:
                sorted_tokens = sorted_tokens[0:kwargs['num_words']]
            sorted_cluster_labels[topic] = sorted_tokens
        return sorted_cluster_labels


