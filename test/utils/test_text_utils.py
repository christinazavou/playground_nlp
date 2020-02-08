import unittest


class TestTextUtilsMethods(unittest.TestCase):

    def test_evaluate_value_or_dict(self):
        from src.utils.text_utils import evaluate_value_or_dict
        self.assertEqual('kati', evaluate_value_or_dict("kati"))
        self.assertEqual({'a': 1}, evaluate_value_or_dict("{'a': 1}"))
        self.assertEqual(4, evaluate_value_or_dict(4))

    def test_is_ascii(self):
        from src.utils.text_utils import is_ascii
        self.assertEqual(True, is_ascii("4"))
        self.assertEqual(False, is_ascii("¶"))

    def test_dictionary_to_utf(self):
        from src.utils.text_utils import dictionary_to_utf
        init_dict = {'a': 1, 'b': 2}
        str_dict = dictionary_to_utf(init_dict)
        self.assertTrue(str_dict in ["{\"a\": 1, \"b\": 2}", "{\"b\": 2, \"a\": 1}"])

    def test_dict_to_str_to_dict(self):
        from src.utils.text_utils import dictionary_to_utf
        from src.utils.text_utils import evaluate_value_or_dict
        init_dict = {'a': 1, 'b': 2}
        str_dict = dictionary_to_utf(init_dict)
        dict_from_str = evaluate_value_or_dict(str_dict)
        self.assertEqual(init_dict, dict_from_str)

    def test_variable_to_utf8(self):
        from src.utils.text_utils import variable_to_utf8
        self.assertEqual(b"1", variable_to_utf8(1))
        self.assertEqual([b"3", b"3"], variable_to_utf8(["3", 3]))
        self.assertEqual([b"something", b"3"], variable_to_utf8(["something", 3]))
        self.assertEqual({b"something":b"1", b"3":b"3"}, variable_to_utf8({"something":1, 3:3}))

    def test_split_sentences(self):
        from src.utils.text_utils import split_sentences
        self.assertEqual(["first sentence", "Second sentence!Still second sentence.", "Third sentence?Yes!"],
                         split_sentences("first sentence   Second sentence!Still second sentence. Third sentence?Yes!"))

    def test_get_uni_grams_and_bi_grams_from_tokens(self):
        from src.utils.text_utils import get_uni_grams_and_bi_grams_from_tokens
        self.assertEqual(["one", "two", "three", "one_two", "two_three"],
                         get_uni_grams_and_bi_grams_from_tokens(["one", "two", "three"]))

    def test_add_uni_grams_and_bi_grams_from_tokens(self):
        from src.utils.text_utils import add_uni_grams_and_bi_grams_from_tokens
        uni_grams_bi_grams = set()
        expected = {"one", "two", "one_two", "two_three", "three", "four", "three_four"}

        uni_grams_bi_grams = add_uni_grams_and_bi_grams_from_tokens(uni_grams_bi_grams, ["one", "two", "three"])
        uni_grams_bi_grams = add_uni_grams_and_bi_grams_from_tokens(uni_grams_bi_grams, ["three", "four"])

        self.assertEqual(expected, uni_grams_bi_grams)

    def test_de_bi_gram(self):
        from src.utils.text_utils import de_bi_gram
        self.assertEqual("one two", de_bi_gram("one_two"))

if __name__ == '__main__':
    unittest.main()

