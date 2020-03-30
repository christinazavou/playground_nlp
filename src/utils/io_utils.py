
# -*- coding: utf-8 -*-

import gzip
import json
import logging
import os
import pandas as pd
import pickle

from src.utils.logger_utils import get_logger

LOGGER = get_logger(__name__, logging.WARNING)


def read_df(filename, chunk_size=None, read_columns=None):
    """
    :return pandas DataFrame
    """
    if '.csv' not in filename:
        LOGGER.warning("The input filename {} might be wrong.".format(filename))
    return pd.read_csv(filename, encoding='utf8', index_col=0, usecols=read_columns, chunksize=chunk_size)


def write_df(df, filename, index=True, header=True, mode='w'):
    """
    NOTE: if you want to save a compressed format use the extension '.gzip'.
    :param df: pandas DataFrame
    :param filename: string
    :param index: boolean
    :param header: boolean
    :param mode: char
    """
    if '.csv' not in filename:
        LOGGER.warning("Consider including .csv in your filename {}".format(filename))
    df.to_csv(filename, encoding='utf8', header=header, index=index, mode=mode)


def _load_gzip_json(filename):
    if '.json' in filename and '.gzip' in filename:
        with gzip.GzipFile(filename, 'r') as fin:
            dictionary = json.loads(fin.read().decode('utf-8'))
            assert isinstance(dictionary, type({})), 'object read in {} is not a dictionary'.format(filename)
            return dictionary
    else:
        raise Exception('Non accepted json gzip file {}'.format(filename))


def _dump_gzip_json(dictionary, filename):
    if '.json' in filename and '.gzip' in filename:
        with gzip.GzipFile(filename, 'w') as fout:
            fout.write(json.dumps(dictionary).encode('utf-8'))
    else:
        raise Exception('Non accepted json gzip file {}'.format(filename))


def read_json(filename):
    if '.gzip' in filename:
        return _load_gzip_json(filename)
    return json.load(open(filename, 'r'))


def write_json(data, filename):
    """
    NOTE: if you want to save a compressed file use the extension '.gzip'
    :param data: dict
    :param filename: string
    """
    if '.gzip' in filename:
        _dump_gzip_json(data, filename)
    else:
        json.dump(data, open(filename, 'w'), ensure_ascii=False, indent=2)


def read_pickle(filename):
    """
    :param filename:
    :return:
    :raises FileNotFoundError
    """
    if '.gzip' in filename:
        f = gzip.GzipFile(filename, 'rb')
        obj = pickle.load(f)
    else:
        assert '.pkl' in filename
        with open(filename, 'rb') as input_data:
            obj = pickle.load(input_data)
    return obj


def write_pickle(obj, filename):
    if '.gzip' in filename:
        f = gzip.GzipFile(filename, 'wb')
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    else:
        assert '.pkl' in filename
        with open(filename, 'wb') as output:
            pickle.dump(obj, output, pickle.HIGHEST_PROTOCOL)


def file_exists(filename):
    return os.path.isfile(filename)
