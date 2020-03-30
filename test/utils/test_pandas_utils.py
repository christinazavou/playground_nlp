import pandas as pd
import unittest


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

    def test_get_preprocessed_text(self):
        from src.utils.pandas_utils import get_preprocessed_text
        senteces_list = [[u'mon', u'spectrum', u'free'], [u'hostnam', u'ram', u'free', u'swap']]

        expected = "mon spectrum free hostnam ram free swap"
        self.assertEqual(expected, get_preprocessed_text(senteces_list, use_bi_grams=False, as_list=False))

        expected = "mon spectrum free mon_spectrum spectrum_free hostname ram free swap hostnam_ram ram_free free_swap"
        expected = list(set(expected.split(" ")))
        self.assertEqual(len(expected), len(get_preprocessed_text(senteces_list, True, False).split(" ")))

        expected = ["mon", "spectrum", "free", "mon_spectrum", "spectrum_free", "hostnam", "ram", "free", "swap", "hostnam_ram", "ram_free", "free_swap"]
        expected = list(set(expected))
        self.assertEqual(len(expected), len(get_preprocessed_text(senteces_list, True, True)))

        expected = ['mon', u'spectrum', u'free', u'hostnam', u'ram', u'free', u'swap']
        self.assertEqual(expected, get_preprocessed_text(senteces_list, False, True))

    def test_iter_tickets_on_field(self):
        from src.utils.pandas_utils import iter_tickets_on_field
        generator = iter_tickets_on_field("textpreprocessed", input_file="../.resources/exampleData.csv",
                                          use_bi_grams=False, as_list=False)
        idx, ticket = next(generator)
        self.assertTrue(isinstance(ticket, str))

        generator = iter_tickets_on_field("textpreprocessed", input_file="../.resources/exampleData.csv",
                                          use_bi_grams=False, as_list=True)
        idx, ticket = next(generator)
        self.assertTrue(isinstance(ticket, list))

if __name__ == '__main__':
    unittest.main()
