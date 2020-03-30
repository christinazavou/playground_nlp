import unittest


class TestDoc2VecUtilsMethods(unittest.TestCase):

    def test_something(self):
        from src.preprocessing.Doc2Vec import CorpusToDoc2Vec
        corpus_to_doc_2_vec = CorpusToDoc2Vec("../.resources/exampleData.csv",
                                              "textpreprocessed",
                                              "../.resources/example_D2V.pkl")

        vectors_matrix = corpus_to_doc_2_vec.get_vectors_as_np()
        self.assertEqual((10, 400), vectors_matrix.shape)

        sent_60540 = corpus_to_doc_2_vec.model.docvecs['SENT_60540']
        self.assertEqual(400, len(sent_60540))

        similar_to_60540 = corpus_to_doc_2_vec.show_similar_documents('SENT_60540', 3)
        print("similar_to_60540: ", similar_to_60540)

        similarity_of_equal = corpus_to_doc_2_vec.similarity_given_indices('SENT_60540', 'SENT_60540')
        print("similarity_of_equal:", similarity_of_equal)

if __name__ == '__main__':
    unittest.main()

    # similar_by_vector(vector, topn=10, restrict_vocab=None)
    # similarity_unseen_docs(model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5)
