
# -*- coding: utf-8 -*-

import json
import pickle
import gzip
import pandas as pd


def load_zip_df(filename, columns=None):
    if '.p' in filename:
        f = gzip.GzipFile(filename, 'rb')

        # for compatibility with Python2 and Python3:
        u = pickle._Unpickler(f)
        u.encoding = 'latin1'
        df = u.load()

        assert isinstance(df, type(pd.DataFrame())), 'object read in {} is not a pandas Dataframe'.format(filename)
        f.close()
        if columns:
            return df[columns]
        return df
    else:
        raise Exception('non accepted zip file {}'.format(filename))


def store_zip_df(df, filename):
    if '.p' in filename and '.zip' in filename:
        f = gzip.GzipFile(filename, 'wb')
        pickle.dump(df, f)
        f.close()
    else:
        raise Exception('non accepted zip file {}'.format(filename))


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


def store_pickle(obj, filename, protocol=None):
    if '.zip' in filename:
        f = gzip.GzipFile(filename, 'wb')
        if protocol:
            pickle.dump(obj, f, protocol)
        else:
            pickle.dump(obj, f)
        f.close()
    else:
        if protocol:
            pickle.dump(obj, open(filename, 'wb'), protocol)
        else:
            pickle.dump(obj, open(filename, 'wb'))


def load_pickle(filename):
    if '.zip' in filename:
        f = gzip.GzipFile(filename, 'rb')
        obj = pickle.load(f)
    else:
        obj = pickle.load(open(filename, 'rb'))
    return obj


def read_df(df_file, chunk_size=None, read_columns=None):
    if '.csv' in df_file:
        if chunk_size:
            return pd.read_csv(df_file, encoding='utf8', index_col=0, chunksize=chunk_size)
        else:
            if read_columns:
                return pd.read_csv(df_file, encoding='utf8', index_col=0, usecols=read_columns)
            else:
                return pd.read_csv(df_file, encoding='utf8', index_col=0)
    elif '.p' in df_file:
        if 'zip' in df_file:
            return load_zip_df(df_file, read_columns)
        if read_columns:
            return pd.read_pickle(df_file)[read_columns]
        else:
            return pd.read_pickle(df_file)
    else:
        raise Exception(' unknown pandas file {}'.format(df_file))


def store_df(df, df_file, index=True, header=True, mode='w'):
    if '.csv' in df_file:
        df.to_csv(df_file, encoding='utf8', header=header, index=index, mode=mode)
    elif '.p' in df_file:
        if 'zip' in df_file:
            store_zip_df(df, df_file)
        else:
            df.to_pickle(df_file)
    else:
        raise Exception(' unknown pandas file {}'.format(df_file))


def read_json(json_file):
    if 'zip' in json_file:
        return load_zip_json(json_file)
    return json.load(open(json_file, 'r'), encoding='utf8')


def store_json(data, json_file):
    if 'zip' in json_file:
        store_zip_json(data, json_file)
    else:
        json.dump(data, open(json_file, 'w'), encoding='utf8', indent=2)


if __name__ == '__main__':
    df = read_df("/media/christina/Elements/Thesis_stuff/nlp-keywords-cupenya/data/LibertyGlobal/DBSCAN/config1/data_frame_selectedpreprocessed.p.zip")
    res = df["textpreprocessed"].iloc[0:10]
    res.to_csv("exampleData.csv", header=True)
