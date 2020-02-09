import unittest


class TestDoc2VecUtilsMethods(unittest.TestCase):

    def test_something(self):
        from src.preprocessing.Doc2Vec import CorpusToDoc2Vec
        corpusToDoc2Vec = CorpusToDoc2Vec("../exampleData.csv", "textpreprocessed", "exampleModel.p")

        vectors_matrix = corpusToDoc2Vec.get_vectors_as_np()
        self.assertEqual((10, 400), vectors_matrix.shape)

        sent_60540 = corpusToDoc2Vec.model.docvecs['SENT_60540']
        self.assertEqual(400, len(sent_60540))

        similar_to_60540 = corpusToDoc2Vec.show_similar('SENT_60540', 3)
        print(similar_to_60540)

        similarity_of_equal = corpusToDoc2Vec.similarity_given_indices('SENT_60540', 'SENT_60540')
        print(similarity_of_equal)

if __name__ == '__main__':
    unittest.main()

    # similar_by_vector(vector, topn=10, restrict_vocab=None)
    # similarity_unseen_docs(model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5)
