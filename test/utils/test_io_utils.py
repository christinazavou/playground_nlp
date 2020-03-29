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

    def test_load_pickle(self):
        from src.utils.io_utils import load_pickle
        try:
            load_pickle("../.resources/exampleModel.p1")
        except FileNotFoundError as e:
            self.assertTrue(True)
        example_model = load_pickle("../.resources/exampleModel.p")
        from gensim.models.doc2vec import Doc2Vec
        self.assertTrue(isinstance(example_model, Doc2Vec))

    def test_read_df(self):
        from src.utils.io_utils import read_df

        df = read_df("../.resources/df_preprocessed.p")
        self.assertTrue(df.shape[0] == 1000)

        df = read_df("../.resources/df_preprocessed_automatic.p.zip")
        self.assertTrue(df.shape[0] == 1000)

        with self.assertRaises(OSError):
            df = read_df("../.resources/df_preprocessed_manual.p.zip")

        with self.assertRaises(NotImplementedError):
            df = read_df("../.resources/df_preprocessed_automatic")

    def test_store_df(self):
        from src.utils.io_utils import store_df, read_df
        df = read_df("../.resources/df_preprocessed.p")

        store_df(df, os.path.join(self.test_dir, 'tmp_df.csv'))
        store_df(df, os.path.join(self.test_dir, 'tmp_df.p'))
        store_df(df, os.path.join(self.test_dir, 'tmp_df.p.zip'))


if __name__ == '__main__':
    unittest.main()

