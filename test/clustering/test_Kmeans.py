import unittest

from src.clustering.Kmeans import KMeansClusterParameters, KMeansModel, ClusterParameters


class TestKMeans(unittest.TestCase):
    def test_save_fitted_labels_parallel(self):
        cluster_params = ClusterParameters(num_clusters=50, use_bi_grams=True, max_df=0.5, min_df=5, max_features=1000)
        kmeans_params = KMeansClusterParameters(vector_size=100, window=5, std_scale=True, svd_n=None, pca_n=None)

        m = KMeansModel(
            '../.resources/example_preprocessed.csv.gzip',
            u'textpreprocessed',
            '../.resources/ALLKMEANS/C1',
            '../.resources/example_stw_text.p.gzip',
            'KMeansCluster',
            cluster_params,
            kmeans_params)

        m.save_fitted_labels_as_list_parallel()


if __name__ == '__main__':
    from multiprocessing import freeze_support

    freeze_support()
    unittest.main()

    # m.save_fitted_labels_parallel(workers=2, chunksize=5000, show_distr=True, show_degree=True)
    #
    # m.evaluate(metric='silhouette', show_distr=True, show_degree=True)
    #
    # m.evaluate(metric='cohesion')
    # logging.info(
    #     'average cohesion = {}'.format(sum(m.cohesion_per_topic.values()) / float(len(m.cohesion_per_topic.values()))))
    #
    # m.find_clusters_and_text(clusters_file, clusters_text_file, degree_field='degree{}'.format(m.name), workers=2)
    #
    # m.save_clusters_labels_json(clusters_file, with_degree=True, use_stw=True, split=True, num_words=7)
    #
    # append_to_json_clusters('cohesion', dict_to_utf(m.cohesion_per_topic), clusters_dict_file=clusters_file)
    #
