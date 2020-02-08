import unittest


class TestDoc2VecUtilsMethods(unittest.TestCase):

    def test_something(self):
        from src.preprocessing.Doc2Vec import CorpusToDoc2Vec
        corpusToDoc2Vec = CorpusToDoc2Vec("exampleData.csv", "textpreprocessed", "exampleModel.p")
        print("aha")

if __name__ == '__main__':
    unittest.main()
