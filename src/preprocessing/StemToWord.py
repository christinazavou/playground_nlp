# -*- coding: utf-8 -*-
import logging
from collections import defaultdict
from src.utils.io_utils import store_pickle, load_pickle
from src.utils.logger_utils import get_logger

logging.root.level = logging.INFO


class StemToWordFileMissingError(Exception):

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class StemNotInStemToWord(Exception):

    def __init__(self, expression, message):
        self.expression = expression
        self.message = message


class StemToWord:

    """
    Stems are normalized words.

    An object of this class can be used throughout preprocessing of many texts ..
    so it is updated with every call ..
    """

    LOGGER = get_logger('StemToWord', logging.INFO)

    words_by_stem_map = {}  # { stem1: { word1 : count1, word2: count2 }, stem2: { word3: count3 } }
    filename = None

    def __init__(self, stw_file=None):
        if stw_file:
            self.filename = stw_file
            try:
                copy_instance = load_pickle(stw_file)
                self.words_by_stem_map = copy_instance.words_by_stem_map
            except FileNotFoundError:
                pass

    def __call__(self, word, stemmer=None):
        if stemmer:
            stem = stemmer.stem(word)
        else:
            stem = word
        self.words_by_stem_map.setdefault(stem, defaultdict(int))
        self.words_by_stem_map[stem][word] += 1
        return stem

    def save(self):
        if self.filename:
            store_pickle(self, self.filename)
        else:
            raise StemToWordFileMissingError("if self.filename", "filename is not set. Please provide that")

    def find_word_from_stem(self, stem):
        try:
            possible_words_dict = self._get_possible_words_from_stem(stem)
            possible_words_counts = [(key, value) for key, value in possible_words_dict.items()]
            sorted_counts = sorted(possible_words_counts, key=lambda x: x[1], reverse=True)
            return sorted_counts[0][0]
        except StemNotInStemToWord:
            return stem

    def _get_possible_words_from_stem(self, stem):
        if stem not in self.words_by_stem_map:
            raise StemNotInStemToWord("stem not in self.words_by_stem_map", u'stem {} not in STW'.format(stem))
        return self.words_by_stem_map[stem]

    # def add_words_to_codes(self, words_to_codes, save=True):
    #     for key_i, key in enumerate(self.words_by_stem_map.keys()):
    #         values = self.words_by_stem_map[key]
    #         for value_i, value in enumerate(values.keys()):
    #             if value in words_to_codes.keys():
    #                 times_value_in_text = self.words_by_stem_map[key][value]
    #                 times_value_in_codes = sum(words_to_codes[value].values())
    #                 if times_value_in_text == times_value_in_codes:
    #                     # remove existing, add codes
    #                     del self.words_by_stem_map[key][value]
    #                     self.words_by_stem_map[key].update(words_to_codes[value])
    #                 else:
    #                     # keep existing with counts =times_value_in_text-times_value_in_codes, add codes
    #                     self.words_by_stem_map[key][value] = times_value_in_text - times_value_in_codes
    #                     self.words_by_stem_map[key].update(words_to_codes[value])
    #
    #     if save:
    #         self.save()
