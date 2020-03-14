# -*- coding: utf-8 -*-
# import timeit
import os
import logging
from collections import defaultdict
from src.utils.io_utils import store_pickle, load_pickle


logging.root.level = logging.INFO


class StemToWord:

    """
    Stems are normalized words.
    """

    def __init__(self, stw_file=None):
        self.words_by_stem_map = {}
        if stw_file:
            self.filename = stw_file
            if os.path.isfile(stw_file):
                copy_instance = load_pickle(stw_file)
                self.words_by_stem_map = copy_instance.stem_by_word_map

    def save_and_return_stem(self, word, stemmer):
        stem = stemmer.stem(word)
        self.words_by_stem_map.setdefault(stem, defaultdict(int))
        self.words_by_stem_map[stem][word] += 1
        return stem

    def save_and_return_unstemmed(self, word):
        self.words_by_stem_map.setdefault(word, defaultdict(int))
        self.words_by_stem_map[word][word] += 1
        return word

    def save(self):
        if self.filename:
            store_pickle(self, self.filename)
        else:
            raise Exception('unknown self.filename to save the stw object')

    def find_word(self, stem):
        # assume words after pre-process have no '_' in them
        try:
            if '_' in stem:
                stems = stem.split('_')
            else:
                stems = stem.split()
            words = []
            for stem in stems:  # in case a token of LDA i.e. 'pleas del'
                possible_words_dict = self.words_by_stem_map[stem]
                possible_words_counts = [(key, value) for key, value in possible_words_dict.iteritems()]
                sorted_counts = sorted(possible_words_counts, key=lambda x: x[1], reverse=True)
                words.append(sorted_counts[0][0])
            return u' '.join(words)
        except:
            print u'stem {} not in STW'.format(unicode(stem))
            return stem

    def add_words_to_codes(self, words_to_codes, save=True):
        for key_i, key in enumerate(self.words_by_stem_map.keys()):
            values = self.words_by_stem_map[key]
            for value_i, value in enumerate(values.keys()):
                if value in words_to_codes.keys():
                    # print 'words_stems_map[{}][{}] = {}'.format(key, value, self.words_stems_map[key][value])
                    # print 'words_to_codes[{}] = {}'.format(value, words_to_codes[value])
                    times_value_in_text = self.words_by_stem_map[key][value]
                    times_value_in_codes = sum(words_to_codes[value].values())
                    # print 'times in text, in codes = {}, {}'.format(times_value_in_text, times_value_in_codes)
                    if times_value_in_text == times_value_in_codes:
                        # remove existing, add codes
                        del self.words_by_stem_map[key][value]
                        self.words_by_stem_map[key].update(words_to_codes[value])
                        # print 'words_stems_map[{}] replaced with {}'.format(key, self.words_stems_map[key])
                    else:
                        # keep existing with counts =times_value_in_text-times_value_in_codes, add codes
                        self.words_by_stem_map[key][value] = times_value_in_text - times_value_in_codes
                        self.words_by_stem_map[key].update(words_to_codes[value])
                        # print 'words_stems_map[{}] transforms to {}'.format(key, self.words_stems_map[key])
        if save:
            self.save()


if __name__ == '__main__':

    """
    Liberty Global
    ol  :  defaultdict(<type 'int'>, {u'ols': 4})
    og  :  defaultdict(<type 'int'>, {u'ogs': 94})
    od  :  defaultdict(<type 'int'>, {u'ods': 149})
    oc  :  defaultdict(<type 'int'>, {u'ocs': 38})
    ow  :  defaultdict(<type 'int'>, {u'ows': 2})
    or  :  defaultdict(<type 'int'>, {u'ors': 2})

    Send Cloud
    io  :  defaultdict(<type 'int'>, {u'ios': 23})
    in  :  defaultdict(<type 'int'>, {u'ins': 6})
    id  :  defaultdict(<type 'int'>, {u'ids': 46})
    if  :  defaultdict(<type 'int'>, {u'ifs': 6})
    oc  :  defaultdict(<type 'int'>, {u'ocs': 1})
    ot  :  defaultdict(<type 'int'>, {u'ots': 100})

    Marley Spoon
    an  :  defaultdict(<type 'int'>, {u'ans': 31})
    aw  :  defaultdict(<type 'int'>, {u'awful': 1})
    el  :  defaultdict(<type 'int'>, {u'els': 76})
    ei  :  defaultdict(<type 'int'>, {u'eis': 8})
    ea  :  defaultdict(<type 'int'>, {u'eaed': 1})
    ho  :  defaultdict(<type 'int'>, {u'hoed': 22})
    """

    stw = StemToWord('..\..\data\LibertyGlobal\KEYWORDS\config1\data_framestw.p.zip')
    for w, items in stw.words_by_stem_map.iteritems():
        print w, items

