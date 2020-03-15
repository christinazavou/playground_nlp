import unittest

from src.preprocessing.StemToWord import StemToWord, StemToWordFileMissingError
from nltk.stem.snowball import SnowballStemmer


class TestStemToWord(unittest.TestCase):

    english_stemmer = SnowballStemmer('english')

    def test_stems(self):
        stem_to_word_obj = StemToWord()
        stem_to_word_obj("language", self.english_stemmer)
        self.assertTrue(len(stem_to_word_obj.words_by_stem_map) == 1)

    def test_save_exception(self):
        stem_to_word_obj = StemToWord()
        with self.assertRaises(StemToWordFileMissingError):
            stem_to_word_obj.save()

    def test_save_ok(self):
        stem_to_word_obj = StemToWord("../.resources/ExampleSTW.p")
        try:
            stem_to_word_obj.save()
        except Exception as e:
            self.fail("stem_to_word_obj.save() raised Exception unexpectedly!", e)

    def test_get_word_from_stem(self):
        stem_to_word_obj = StemToWord()
        stem_to_word_obj("language", self.english_stemmer)
        self.assertTrue(stem_to_word_obj.find_word_from_stem("languag") == "language")
        self.assertTrue(stem_to_word_obj.find_word_from_stem("moth") == "moth")


if __name__ == '__main__':
    unittest.main()



