
# -*- coding: utf-8 -*-

import gzip
import json
import os
import pickle

import pandas as pd


def _load_pickled_zipped_df(filename):
    if '.p' not in filename or '.zip' not in filename:
        raise NotImplementedError('non accepted pickled zipped filename {}'.format(filename))

    f = gzip.GzipFile(filename, 'rb')

    # for compatibility with Python2 and Python3:
    u = pickle._Unpickler(f)
    u.encoding = 'latin1'
    df = u.load()

    assert isinstance(df, type(pd.DataFrame())), 'object read in {} is not a pandas Dataframe'.format(filename)
    f.close()
    return df


def _store_pickled_zipped_df(df, filename):
    if '.p' not in filename or '.zip' not in filename:
        raise Exception('non accepted zip file {}'.format(filename))
    else:
        f = gzip.GzipFile(filename, 'wb')
        pickle.dump(df, f)
        f.close()


def read_df(df_file, chunk_size=None, read_columns=None):
    if '.p' in df_file:
        if '.zip' in df_file:
            df = _load_pickled_zipped_df(df_file)
        else:
            df = pd.read_pickle(df_file)
        if read_columns:
            return df[read_columns]
        else:
            return df
    if '.csv' in df_file:
        return pd.read_csv(df_file, encoding='utf8', index_col=0, usecols=read_columns, chunksize=chunk_size)
    else:
        raise NotImplementedError('Can\' read pandas DataFrame from filename {}'.format(df_file))


def store_df(df, df_file, index=True, header=True, mode='w'):
    if '.p' in df_file:
        if '.zip' in df_file:
            _store_pickled_zipped_df(df, df_file)
        else:
            df.to_pickle(df_file)
    elif '.csv' in df_file:
        df.to_csv(df_file, encoding='utf8', header=header, index=index, mode=mode)
    else:
        raise NotImplementedError('Can\'t store pandas DataFrame for filename {}'.format(df_file))


def load_zip_json(filename):
    if '.json' in filename and '.zip' in filename:
        dictionary = json.load(gzip.GzipFile(filename, 'r'), encoding='utf8')
        assert isinstance(dictionary, type({})), 'object read in {} is not a dictionary'.format(filename)
        return dictionary
    else:
        raise Exception('non accepted zip file {}'.format(filename))


def store_zip_json(dictionary, filename):
    if '.json' in filename and '.zip' in filename:
        json.dump(dictionary, gzip.GzipFile(filename, 'w'), encoding='utf8', indent=2)
    else:
        raise Exception('non accepted zip file {}'.format(filename))


def read_json(json_file):
    if 'zip' in json_file:
        return load_zip_json(json_file)
    return json.load(open(json_file, 'r'), encoding='utf8')


def store_json(data, json_file):
    if 'zip' in json_file:
        store_zip_json(data, json_file)
    else:
        json.dump(data, open(json_file, 'w'), encoding='utf8', indent=2)


def store_pickle(obj, filename):
    if '.zip' in filename:
        f = gzip.GzipFile(filename, 'wb')
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)
        f.close()
    else:
        assert '.p' in filename
        pickle.dump(obj, open(filename, 'wb'), pickle.HIGHEST_PROTOCOL)


def load_pickle(filename):
    """
    :param filename:
    :return:
    :raises FileNotFoundError
    """
    if '.zip' in filename:
        f = gzip.GzipFile(filename, 'rb')
        obj = pickle.load(f)
    else:
        assert '.p' in filename
        obj = pickle.load(open(filename, 'rb'))
    return obj


def file_exists(filename):
    return os.path.isfile(filename)
