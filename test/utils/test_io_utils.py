import unittest


class TestIOUtilsMethods(unittest.TestCase):

    def test_load_pickle(self):
        from src.utils.io_utils import load_pickle
        try:
            load_pickle("../.resources/exampleModel.p1")
        except FileNotFoundError as e:
            self.assertTrue(True)
        example_model = load_pickle("../.resources/exampleModel.p")
        from gensim.models.doc2vec import Doc2Vec
        self.assertTrue(isinstance(example_model, Doc2Vec))

if __name__ == '__main__':
    unittest.main()

