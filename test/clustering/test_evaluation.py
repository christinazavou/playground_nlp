import unittest
from scipy.sparse import csr_matrix

from src.clustering.evaluation import ClusterEvaluator


class TestEvaluation(unittest.TestCase):

    def test_silhouette_coefficient(self):

        x = [
            [10, 20, 30],
            [10, 40, 30],
            [20, 30, 30],
            [10, 25, 50]
        ]

        y = [
            1,
            1,
            1,
            2
        ]
        cluster_evaluator = ClusterEvaluator()
        sc = cluster_evaluator.silhouette_coefficient(x, y)
        self.assertTrue(-1 <= sc <= 1)

    def test_topic_coherence(self):

        corpus = [
            # NUM_DOCS x NUM_WORDS (reversed from gensim corpus representation)
            [1,0,0,0,1],
            [1,1,0,0,0],
            [1,0,0,1,1],
            [1,1,1,1,1],
            [1,1,0,0,0],
            [0,0,1,1,1],
            [0,0,1,0,1]
        ]

        words_ids_per_topic = {
            # topicId: [word1, word10, word2,..]  # cutted list of K top words
            1: [1,3,5],
            2: [2,4,5]
        }

        sparse_corpus = csr_matrix(corpus)

        cluster_evaluator = ClusterEvaluator()
        cluster_evaluator.topics_cohesion_sparse(sparse_corpus, words_ids_per_topic)


if __name__ == '__main__':
    unittest.main()
