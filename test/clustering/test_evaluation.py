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
        # NOTE: docId, topicId and wordId starts from zero

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
            1: [100, 300, 400],
            2: [200, 400]
        }

        sparse_corpus = csr_matrix(corpus)

        cluster_evaluator = ClusterEvaluator()
        cohesion_per_topic = cluster_evaluator.topics_cohesion_sparse(sparse_corpus, [0, 100, 200, 300, 400],
                                                                      words_ids_per_topic)
        [self.assertTrue(-1 <= cohesion_per_topic[topic] <= 1) for topic in words_ids_per_topic.keys()]


if __name__ == '__main__':
    unittest.main()
