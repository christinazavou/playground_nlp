import unittest
import pandas as pd


class TestPandasUtilsMethods(unittest.TestCase):

    def setUp(self):
        dummy_data = ['Geeks', 'For', 'Geeks', 'is', 'portal', 'for', 'Geeks']
        self.df = pd.DataFrame(dummy_data)

    def test_manage_logger(self):
        from src.utils.pandas_utils import chunk_dataframe_serial
        result = chunk_dataframe_serial(self.df, 3)
        result_list = list(result)
        self.assertEqual(3, len(result_list))
        self.assertTrue(isinstance(result_list[0], pd.DataFrame))
        self.assertTrue(isinstance(result_list[1], pd.DataFrame))
        self.assertTrue(isinstance(result_list[2], pd.DataFrame))
        self.assertEqual(3, len(result_list[0]))
        self.assertEqual(3, len(result_list[1]))
        self.assertEqual(1, len(result_list[2]))


if __name__ == '__main__':
    unittest.main()
