# -*- coding: utf-8 -*-

import ast
import json
import copy
import re


def evaluate_value_or_dict(value):
    try:
        # if s is a string representation of a dictionary then return the dictionary
        # for example return {'a': 1, 'b': 2} if the input string is "{'a': 1, 'b': 2}"
        return ast.literal_eval(value)
    except Exception:
        return value


def is_ascii(value):
    return all(ord(c) < 128 for c in value)


def dictionary_to_utf(dictionary):
    return json.dumps(dictionary)


def variable_to_utf8(value):

    if isinstance(value, list):
        return [variable_to_utf8(x) for x in value]

    if isinstance(value, dict):
        new_dict = dict()
        for key, value in value.items():
            new_dict[variable_to_utf8(key)] = variable_to_utf8(copy.deepcopy(value))
        return new_dict

    if isinstance(value, str):
        return value.encode('utf-8')

    elif isinstance(value, int) or isinstance(value, float):
        return str(value).encode('utf-8')

    elif isinstance(value, tuple):
        return variable_to_utf8(value[0]), variable_to_utf8(value[1])

    else:
        print("s: ", value)
        print("t: ", type(value))
        raise Exception("unknown type to encode ...")


def split_sentences(text):
    # splits into two sentences if it encounters a "." or "!" or "?" followed by a whitespace
    # or if it encounters more than one whitespaces
    sentences = []
    for sentence in re.split(u'(?<=[^A-Z].[.!?]) +(?=[A-Z])', text, flags=re.U):
        for s in re.split(u'\s{2,}', sentence, flags=re.U):
            sentences.append(s)
    return sentences


def get_uni_grams_and_bi_grams_from_tokens(sentence_tokens):
    uni_grams_bi_grams = []
    uni_grams_bi_grams += sentence_tokens
    if len(sentence_tokens) > 1:
        for i in range(0, len(sentence_tokens)-1):
            uni_grams_bi_grams += [u'{}_{}'.format(sentence_tokens[i], sentence_tokens[i+1])]
    return uni_grams_bi_grams


def add_uni_grams_and_bi_grams_from_tokens(uni_grams_bi_grams, sentence_tokens):
    assert isinstance(uni_grams_bi_grams, set), "Wrong input type for uni_grams_bi_grams"
    uni_grams_bi_grams |= set(sentence_tokens)
    if len(sentence_tokens) > 1:
        for i in range(0, len(sentence_tokens)-1):
            uni_grams_bi_grams |= {u'{}_{}'.format(sentence_tokens[i], sentence_tokens[i + 1])}
    return uni_grams_bi_grams


def de_bi_gram(token):
    return token.replace(u'_', u' ')

