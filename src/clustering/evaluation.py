import math

import numpy as np
from scipy.sparse import lil_matrix
from sklearn.metrics import silhouette_score
from sklearn.metrics.pairwise import pairwise_distances


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
    def topics_cohesion_sparse(sparse_corpus, vocabulary_ids, words_ids_per_topic):
        """
        :param sparse_corpus: sparse matrix of shape (docsL, wordsL)
        :param words_ids_per_topic: dictionary of format {topic1: [w1, w2,..], topic2: []..}.
        :param vocabulary_ids: list of ids of words with the order appeared in corpus
        :return:
        """

        # Get a list with all words_ids appearing in all topics
        union_topics_words_ids_list = list(set().union(*words_ids_per_topic.values()))

        # Get a sparse matrix representing the occurrences of each topic word in each document
        cm_sparse = _counts_matrix_lil(sparse_corpus, vocabulary_ids, union_topics_words_ids_list)

        cohesion_per_topic = {}
        for topic, words_ids in words_ids_per_topic.items():
            if len(words_ids) == 0:
                cohesion_per_topic[topic] = -1
            else:
                cohesion_per_topic[topic] = _topic_cohesion_sparse(_topic_npmi_matrix_sparse(cm_sparse,
                                                                                             union_topics_words_ids_list,
                                                                                             words_ids))
        return cohesion_per_topic


def _counts_matrix_sparse(sparse_corpus, topics_words_ids_list):
    # words_counts_in_corpus = csr_matrix((sparse_corpus.shape[0], len(topics_words_ids_list)), dtype=np.int)
    words_counts_in_corpus = lil_matrix((sparse_corpus.shape[0], len(topics_words_ids_list)), dtype=np.int)
    idx_x, idx_y = sparse_corpus.nonzero()
    for x, y in zip(idx_x, idx_y):
        if y in topics_words_ids_list:
            i = topics_words_ids_list.index(y)
            words_counts_in_corpus[x, i] = 1
    return words_counts_in_corpus


def _counts_matrix_lil(corpus_sparse_mat, vocabulary_ids, union_topics_words_ids_list):
    """
    :param corpus_sparse_mat: sparse matrix of shape Docs x Words
    :param vocabulary_ids: list of ids of words with the order appeared in corpus
    :param union_topics_words_ids_list: list of all words_ids appearing in all topics
    :return: a sparse matrix of shape Docs x TopicWords representing the occurrences of each topic word in each document
    """

    num_of_documents = corpus_sparse_mat.shape[0]
    num_of_all_topic_words = len(union_topics_words_ids_list)

    # Matrix of shape Docs x WordsInTopics (T)
    t_words_counts_in_documents = lil_matrix((num_of_documents, num_of_all_topic_words), dtype=np.int)

    for i in range(num_of_documents):
        for j in range(num_of_all_topic_words):
            index_of_topic_word_in_corpus_mat = vocabulary_ids.index(union_topics_words_ids_list[j])
            if corpus_sparse_mat[i, index_of_topic_word_in_corpus_mat] > 0:
                t_words_counts_in_documents[i, j] = 1

    return t_words_counts_in_documents.tocsc()


def _topic_npmi_matrix_sparse(t_words_counts_in_corpus, union_topics_words_ids_list, words_ids_in_a_topic_list):
    """
    :param t_words_counts_in_corpus: a sparse matrix of shape Docs x TopicWords representing the occurrences of each topic word in each document
    :param union_topics_words_ids_list: list of all words_ids appearing in all topics
    :param words_ids_in_a_topic_list: list of all words_ids appearing in specific topic
    :return: matrix of shape TopicWords x TopicWords. These are the coherence scores
    """
    npmi_matrix = np.zeros((len(words_ids_in_a_topic_list), len(words_ids_in_a_topic_list)))
    for i, word_id_i in enumerate(words_ids_in_a_topic_list):
        idx_i = union_topics_words_ids_list.index(word_id_i)
        for j, word_id_j in enumerate(words_ids_in_a_topic_list):
            if j < i:
                npmi_matrix[i, j] = npmi_matrix[j, i]
            else:
                idx_j = union_topics_words_ids_list.index(word_id_j)
                npmi_matrix[i, j] = _npmi_sparse(t_words_counts_in_corpus, idx_i, idx_j)
    return npmi_matrix


def _npmi_sparse(words_counts_in_corpus, idx_i, idx_j):
    """
    :param words_counts_in_corpus: a sparse matrix of shape Doc x Word representing the occurrences of each word in each document
    :param idx_i: index of a word
    :param idx_j: index of a word
    :return: float. coherence?cohesion? score of the two input words
    """
    num_of_documents = words_counts_in_corpus.shape[0]

    # in how many docs wi and wj both appear / docs in total
    p_wi_wj = (
                  words_counts_in_corpus[0:, idx_i].T.dot(
                      words_counts_in_corpus[0:, idx_j]).sum() + 1.0) / num_of_documents

    # in how many docs wi appears / docs in total
    p_wi = (np.sum(words_counts_in_corpus[0:, idx_i]) + 1.0) / num_of_documents

    # in how many docs wj appears / docs in total
    p_wj = (np.sum(words_counts_in_corpus[0:, idx_j]) + 1.0) / num_of_documents

    pmi_i_j = p_wi_wj / (p_wi * p_wj)

    # smoothing
    npmi_i_j = math.log(pmi_i_j, 2) / (-math.log(p_wi_wj, 2))

    return npmi_i_j


def _topic_cohesion_sparse(npmi_matrix):
    """
    :param npmi_matrix: matrix of shape TopicWords x TopicWords. These are the coherence scores
    :return: float. This is cohesion overall topics
    """
    n = npmi_matrix.shape[0]
    cosine_distances = pairwise_distances(npmi_matrix, metric='cosine')
    cosine_similarities = 1.0 - cosine_distances
    cosine_similarities = np.multiply(np.tril(np.ones((n, n)), k=-1), cosine_similarities)
    similarity = np.sum(cosine_similarities)
    combinations = float(math.factorial(n)) / (math.factorial(2)*math.factorial(n-2))
    cohesion = similarity / combinations
    return cohesion
