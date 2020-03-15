import os
import time
import logging
import pandas as pd
from scipy import sparse
import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import Normalizer
from sklearn.pipeline import make_pipeline
from src.utils.pandas_utils import iter_tickets_on_field
from sklearn.preprocessing import StandardScaler
from src.preprocessing.Doc2Vec import CorpusToDoc2Vec
from sklearn.feature_extraction.text import TfidfVectorizer
from .ClusteringAlgorithms import ClusterModel
from src.utils.io_utils import read_df
from src.utils.logger_utils import get_logger


class Kmeans_Model(ClusterModel):

    def __init__(self, data_file, field, model_file, stw_file, name, num_clusters=30, use_bi_grams=False,
                 max_df=0.75, min_df=5, max_features=None,
                 vec_size=None, window=None, std_scale=True, svd_n=None, pca_n=None):

        super(Kmeans_Model, self).__init__(data_file, field, model_file, stw_file, name, num_clusters, use_bi_grams)
        self.max_features = max_features

        logger = manage_logger(__name__, 'INFO', os.path.dirname(os.path.realpath(model_file)), 'train.log')

        if not os.path.isfile(model_file):

            if vec_size and window:
                doc2vec = CorpusToDoc2Vec(data_file, field, model_file.replace('.p', 'd2v.p'), vec_size=vec_size,
                                          window=window)
                self.corpus = doc2vec.get_vectors_as_np()
            else:
                doc_stream = (text for _, text in iter_tickets_on_field(field, datafile=data_file, use_bi_grams=use_bi_grams, as_list=False))
                vectorizer = TfidfVectorizer(max_df=max_df, min_df=min_df, norm='l2', max_features=max_features)
                self.corpus = vectorizer.fit_transform(doc_stream)

            if svd_n:
                svd = TruncatedSVD(n_components=svd_n, random_state=40)
                self.corpus = svd.fit_transform(self.corpus)

            if type(self.corpus) is np.ndarray:
                self.corpus = sparse.csr_matrix(self.corpus)
            logging.info('corpus shape: {}'.format(self.corpus.shape))

            if std_scale:
                self.corpus = StandardScaler(with_mean=False).fit_transform(self.corpus)
                self.corpus = sparse.csr_matrix(self.corpus)

            self.corpus = self.corpus.astype(np.float64)
            s_time = time.time()
            self.model = KMeans(n_clusters=num_clusters, random_state=100)
            cluster_labels = self.model.fit_predict(self.corpus)
            logging.info('Kmeans on {} samples finished after {} seconds'.format(self.corpus.shape, time.time() - s_time))

            cluster_labels = pd.DataFrame({'Label': cluster_labels})
            for cluster, docs in cluster_labels.groupby('Label'):
                print 'cluster {} has {} documents'.format(cluster, docs.shape[0])

            exit()
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
    data_file_ = '..\..\data\LibertyGlobal\DBSCAN\config1\data_frame_selectedpreprocessed.p.zip'
    field_ = u'textpreprocessed'
    model_file_ = '..\..\models\LibertyGlobal\DBSCAN\config1\DBSCAN10_mod_text.p.zip'
    stw_file_ = '..\..\data\LibertyGlobal\DBSCAN\config1\stw_text.p.zip'
    name_ = 'DBSCAN10_mod_'
    use_bi_grams_ = True
    max_df_ = 0.5
    min_df_ = 5
    max_features_ = 10000

    m = Kmeans_Model(data_file_, field_, model_file_, stw_file_, name_, 300,
                     use_bi_grams_, max_df_, min_df_, max_features_,
                     500, 10, True, None, None)
                     # None, None, True, None, None)


