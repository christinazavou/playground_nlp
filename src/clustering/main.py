from multiprocessing import freeze_support

from src.clustering.Kmeans import ClusterParameters, KMeansModel, KMeansClusterParameters

if __name__ == '__main__':
    freeze_support()

    # df = read_df('../.resources/example_preprocessed.csv.gzip')
    # print(df.columns)
    # del df['clusterKMeansCluster']
    # write_df(df, '../.resources/example_preprocessed.csv.gzip')
    # exit()

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

    m.save_fitted_labels_as_list_parallel(chunksize=500)
