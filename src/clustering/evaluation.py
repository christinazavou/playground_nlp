import math
import numpy as np
from scipy.sparse import csr_matrix, lil_matrix
from sklearn.metrics.pairwise import pairwise_distances
from sklearn.metrics import silhouette_score


class ClusterEvaluator(object):

    def __init__(self):
        pass

    @staticmethod
    def silhouette_coefficient(input_data, labels):
        """
        Uses the cosine similarity for fast evaluation (a slower option is jaccard)

        :param input_data:
        :param labels:
        :return:
        """
        return silhouette_score(input_data, labels, metric='cosine', random_state=40)

    @staticmethod
    def topics_cohesion_sparse(sparse_corpus, words_ids_per_topic):
        """ dense_corpus = (docsL, wordsL)
            words_ids_per_topic = {topic1: [w1, w2,..], topic2: []..} """

        # Get a list with all words_ids appearing in all topics
        union_topics_words_ids_list = list(set().union(*words_ids_per_topic.values()))

        # Get a sparse matrix representing the occurrences of each topic word in each document
        cm_sparse = counts_matrix_lil(sparse_corpus, union_topics_words_ids_list)

        cohesion_per_topic = {}
        for topic, words_ids in words_ids_per_topic.items():
            if len(words_ids) == 0:
                cohesion_per_topic[topic] = -1
            else:
                cohesion_per_topic[topic] = topic_cohesion_sparse(topic_npmi_matrix_sparse(cm_sparse,
                                                                                           union_topics_words_ids_list,
                                                                                           words_ids))
        return cohesion_per_topic


def counts_matrix_sparse(sparse_corpus, topics_words_ids_list):
    # words_counts_in_corpus = csr_matrix((sparse_corpus.shape[0], len(topics_words_ids_list)), dtype=np.int)
    words_counts_in_corpus = lil_matrix((sparse_corpus.shape[0], len(topics_words_ids_list)), dtype=np.int)
    idx_x, idx_y = sparse_corpus.nonzero()
    for x, y in zip(idx_x, idx_y):
        if y in topics_words_ids_list:
            i = topics_words_ids_list.index(y)
            words_counts_in_corpus[x, i] = 1
    return words_counts_in_corpus


def counts_matrix_lil(corpus_sparse_mat, union_topics_words_ids_list):
    """
    :param corpus_sparse_mat: sparse matrix of shape Docs x Words
    :param union_topics_words_ids_list: list of all words_ids appearing in all topics
    :return: a sparse matrix of shape Docs x TopicWords representing the occurrences of each topic word in each document
    """

    num_of_documents = corpus_sparse_mat.shape[0]
    num_of_all_topic_words = len(union_topics_words_ids_list)

    # Matrix of shape Docs x WordsInTopics
    words_counts_in_documents = lil_matrix((num_of_documents, num_of_all_topic_words), dtype=np.int)

    for i in range(num_of_documents):
        for j in range(num_of_all_topic_words):
            if corpus_sparse_mat[i, union_topics_words_ids_list[j]] > 0:
                words_counts_in_documents[i, j] = 1

    return words_counts_in_documents.tocsc()


def topic_npmi_matrix_sparse(words_counts_in_corpus, topics_words_ids_list, topic_words_ids_list):
    npmi_matrix = np.zeros((len(topic_words_ids_list), len(topic_words_ids_list)))
    documents = words_counts_in_corpus.shape[0]
    for i, word_id_i in enumerate(topic_words_ids_list):
        idx_i = topics_words_ids_list.index(word_id_i)
        for j, word_id_j in enumerate(topic_words_ids_list):
            if j < i:
                npmi_matrix[i, j] = npmi_matrix[j, i]
            else:
                idx_j = topics_words_ids_list.index(word_id_j)
                npmi_matrix[i, j] = npmi_sparse(words_counts_in_corpus, idx_i, idx_j, documents)
    return npmi_matrix


def npmi_sparse(words_counts_in_corpus, idx_i, idx_j, documents):
    """ with smoothing"""
    p_wi_wj = (words_counts_in_corpus[0:, idx_i].T.dot(words_counts_in_corpus[0:, idx_j]).sum() + 1.0) / documents  # in how many docs wi and wj both appear / docs in total
    p_wi = (np.sum(words_counts_in_corpus[0:, idx_i]) + 1.0) / documents  # in how many docs wi appears / docs in total
    p_wj = (np.sum(words_counts_in_corpus[0:, idx_j]) + 1.0) / documents  # in how many docs wj appears / docs in total
    pmi_i_j = p_wi_wj / (p_wi * p_wj)
    npmi_i_j = math.log(pmi_i_j, 2) / (-math.log(p_wi_wj, 2))
    return npmi_i_j


def topic_cohesion_sparse(npmi_matrix):
    n = npmi_matrix.shape[0]
    cosine_distances = pairwise_distances(npmi_matrix, metric='cosine')
    cosine_similarities = 1.0 - cosine_distances
    cosine_similarities = np.multiply(np.tril(np.ones((n, n)), k=-1), cosine_similarities)
    similarity = np.sum(cosine_similarities)
    combinations = float(math.factorial(n)) / (math.factorial(2)*math.factorial(n-2))
    cohesion = similarity / combinations
    return cohesion
