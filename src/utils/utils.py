from __future__ import unicode_literals

import os
import math
import logging
import numpy as np


def to_remove_indices(df, condition_dict, logger=None):
    """
    Keep rows that have at least one unsatisfied condition in the condition_dict, since conditions define an empty field
    """
    if not condition_dict:
        return df
    c_sat = False
    for field, condition in condition_dict.iteritems():
        cases_sat = df[field] != condition
        c_sat |= cases_sat
        if logger:
            logger.info('{} have {} != {}'.format(np.sum(cases_sat), field, condition))
        else:
            logging.info('{} have {} != {}'.format(np.sum(cases_sat), field, condition))
    df = df[c_sat]
    return df


def iter_tickets_with_degree(field, degree_field, datafile=None, data=None, use_bi_grams=False):
    """The field should contains lists (e.g. a preprocessed field)"""
    if data is None:
        if datafile:
            data = read_df(datafile)
        else:
            raise Exception('nothing to iterate on was given')
    for idx, row in data.iterrows():
        text = eval_utf(row[field])
        degree = row[degree_field]
        if text != []:
            for i in range(int(math.log(degree * 100))):
                text += text
        if not use_bi_grams:
            yield idx, u' '.join([item for sublist in text for item in sublist])
        else:
            yield idx, u' '.join(bi_grams_on_sentences_list(text))


def append_column(df_file, column_name, column_data):
    """no handling for csv params so better give pickle ! """
    df = read_df(df_file)
    df[column_name] = column_data
    store_df(df, df_file)


def rename_column(df_file, in_col, out_col):
    df = read_df(df_file)
    df[out_col] = df[in_col]
    del df[in_col]
    store_df(df, df_file)


def remove_column(df_file, column):
    df = read_df(df_file)
    if column in list(df):
        del df[column]
        store_df(df, df_file)


def pickle_to_csv(pickle_file):  # not zipped support !!!
    pickle_file = pickle_file.replace('.p', '.csv')
    if '.zip' in pickle_file:
        pickle_file = pickle_file.replace('.zip', '')
    return pickle_file


def columns_in_df(df_file, columns):
    df_list = list(read_df(df_file))
    for column in columns:
        if column not in df_list:
            return False
    return True


def delete_file(filename):
    os.remove(filename)


def from_a_process_context_to_a_df(df1, df2, p_c_key, new_field):
    df2[new_field] = u''
    for idx, row in df1.iterrows():
        key_value = eval_utf(row['processContext']).get(p_c_key, '')
        instance = row['instanceId']
        other_idx = df2[df2['instanceId'] == instance].index.tolist()
        if len(other_idx) > 0:
            df2.set_value(other_idx[0], new_field, key_value)
    return df2


def from_df_to_df(df1, df2, key_field, new_field):
    df2[new_field] = ''
    for idx, row in df1.iterrows():
        idx2 = df2[df2['instanceId'] == row['instanceId']].index.tolist()
        if len(idx2) > 0:
            df2.set_value(idx2[0], new_field, row[key_field])
    return df2


def cluster_dict_to_df(cluster_dict, cluster_field, new_field, df=None, df_file=None, key=int, default=''):
    assert df or df_file, 'not enough arguments. give df or df_file.'
    if not df:
        df = read_df(df_file)

    for cluster, cluster_df in df.groupby(cluster_field):
        if not isinstance(cluster, key):
            cluster = key(cluster)
        value = cluster_dict[cluster] if cluster in cluster_dict else default
        for idx, row in cluster_df.iterrows():
            df.set_value(idx, new_field, value)

    if not df_file:
        return df
    store_df(df, df_file)

