import os
import shutil
import tempfile
import unittest


class TestIOUtilsMethods(unittest.TestCase):
    def setUp(self):
        # Create a temporary directory
        self.test_dir = tempfile.mkdtemp()

    def tearDown(self):
        # Remove the directory after the test
        shutil.rmtree(self.test_dir)

    def test_read_write_df(self):
        from src.utils.io_utils import write_df, read_df

        df = read_df("../.resources/example_preprocessed.csv.gzip")
        self.assertTrue(df.shape[0] == 1000)

        df = read_df("../.resources/example_preprocessed.csv.gzip")

        write_df(df, os.path.join(self.test_dir, 'tmp_df.csv'))
        df_in = read_df(os.path.join(self.test_dir, 'tmp_df.csv'))
        self.assertTrue(all(df == df_in))

    def test_read_write_json(self):
        from src.utils.io_utils import read_json, write_json
        test_dict = {'one': ['two', 3]}
        write_json(test_dict, os.path.join(self.test_dir, 'tmp_dict.json'))
        test_dict_in = read_json(os.path.join(self.test_dir, 'tmp_dict.json'))
        self.assertTrue(test_dict == test_dict_in)

        write_json(test_dict, os.path.join(self.test_dir, 'tmp_dict.json.gzip'))
        test_dict_in = read_json(os.path.join(self.test_dir, 'tmp_dict.json.gzip'))
        self.assertTrue(test_dict == test_dict_in)

    def test_read_write_pickle(self):
        from src.utils.io_utils import read_pickle, write_pickle

        with self.assertRaises(FileNotFoundError):
            read_pickle("../.resources/exampleModel.p1")

        example_model = read_pickle("../.resources/exampleModel.p")
        from gensim.models.doc2vec import Doc2Vec
        self.assertTrue(isinstance(example_model, Doc2Vec))

        write_pickle(example_model, os.path.join(self.test_dir, 'tmp_model.p.gzip'))
        example_model_in = read_pickle(os.path.join(self.test_dir, 'tmp_model.p.gzip'))
        self.assertTrue(isinstance(example_model_in, Doc2Vec))


if __name__ == '__main__':
    unittest.main()

