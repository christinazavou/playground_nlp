import os
import time

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import StandardScaler

from src.clustering.ClusteringAlgorithms import ClusterModel, ClusterParameters
from src.preprocessing.Doc2Vec import CorpusToDoc2Vec
from src.utils.pandas_utils import iter_tickets_on_field


class KMeansClusterParameters(object):
    def __init__(self, vector_size=None, window=None, std_scale=True, svd_n=None, pca_n=None):
        self.vector_size = vector_size
        self.window = window
        self.std_scale = std_scale
        self.svd_n = svd_n
        self.pca_n = pca_n


class KMeansModel(ClusterModel):
    def _init_corpus(self):
        if self.kmeans_cluster_parameters.vector_size and self.kmeans_cluster_parameters.window:
            doc2vec = CorpusToDoc2Vec(self.data_file,
                                      self.field,
                                      self.model_file.replace('.p', 'd2v.p'),
                                      vector_size=self.kmeans_cluster_parameters.vector_size,
                                      window=self.kmeans_cluster_parameters.window)
            self.corpus = doc2vec.get_vectors_as_np()
        else:
            doc_stream = (text for _, text in iter_tickets_on_field(self.field,
                                                                    input_file=self.data_file,
                                                                    use_bi_grams=self.cluster_parameters.use_bi_grams,
                                                                    as_list=False))
            vectorizer = TfidfVectorizer(max_df=self.cluster_parameters.max_df,
                                         min_df=self.cluster_parameters.min_df,
                                         norm='l2',
                                         max_features=self.cluster_parameters.max_features)
            self.corpus = vectorizer.fit_transform(doc_stream)

    def _create_corpus(self):
        self._init_corpus()

        if self.kmeans_cluster_parameters.svd_n:
            svd = TruncatedSVD(n_components=self.kmeans_cluster_parameters.svd_n, random_state=40)
            self.corpus = svd.fit_transform(self.corpus)

        if self.kmeans_cluster_parameters.std_scale:
            self.corpus = StandardScaler(with_mean=False).fit_transform(self.corpus)

        self.corpus = self.corpus.astype(np.float64)
        if isinstance(self.corpus, np.ndarray):
            self.corpus = sparse.csr_matrix(self.corpus)
        self.logger.info('Corpus shape: {}'.format(self.corpus.shape))

    def _fit_clusters(self):
        s_time = time.time()
        self.model = KMeans(n_clusters=self.cluster_parameters.num_clusters, random_state=100)
        cluster_labels = self.model.fit_predict(self.corpus)
        self.logger.info(
            'KMeans on {} samples finished after {} seconds'.format(self.corpus.shape, time.time() - s_time))
        return cluster_labels

    def __init__(self, data_file, field, model_file, stw_file, name,
                 cluster_parameters=ClusterParameters(), kmeans_cluster_parameters=KMeansClusterParameters()):

        self.cluster_parameters = cluster_parameters
        self.kmeans_cluster_parameters = kmeans_cluster_parameters
        super(KMeansModel, self).__init__(data_file, field, model_file, stw_file, name, cluster_parameters)

        if not os.path.isfile(model_file):
            self._create_corpus()
            cluster_labels = self._fit_clusters()

            cluster_labels = pd.DataFrame({'Label': cluster_labels})
            for cluster, docs in cluster_labels.groupby('Label'):
                print('cluster {} has {} documents'.format(cluster, docs.shape[0]))

            self.save()
        else:
            self.load()

    def predict(self, docs, **kwargs):
        pass

    def get_vocabulary_tokens(self, split=True, stw=None, use_stw=False):
        pass

    def get_sorted_clusters_labels(self, **kwargs):
        pass

    def get_fitted_labels(self, **kwargs):
        pass


if __name__ == '__main__':
    cluster_params = ClusterParameters(num_clusters=50, use_bi_grams=True, max_df=0.5, min_df=5, max_features=1000)
    kmeans_params = KMeansClusterParameters(vector_size=100, window=5, std_scale=True, svd_n=None, pca_n=None)

    m = KMeansModel(
        '../.resources/df_preprocessed_automatic.p.zip',
        u'textpreprocessed',
        '../.resources/DBSCAN10_mod_text.p.zip',
        '../.resources/stw_text.p.zip',
        'DBSCAN10_mod_',
        cluster_params,
        kmeans_params)
