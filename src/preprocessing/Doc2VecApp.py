from __future__ import unicode_literals

import warnings
warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import os
import time
import json
import logging
import numpy as np
from flask import Flask
from gensim.models import Doc2Vec
from multiprocessing import freeze_support
from src.utils.utils import iter_tickets_on_field, read_df, eval_utf, var_to_utf
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, euclidean


app = Flask(__name__)


class LabeledLineSentence(object):
    def __init__(self, filename, field):
        self.filename = filename
        self.field = field

    def __iter__(self):
        df = read_df(self.filename)
        for idx, row in df.iterrows():
            text = eval_utf(row[self.field])
            tokens = [item for sublist in text for item in sublist]
            yield TaggedDocument(words=tokens, tags=['SENT_%s' % idx])


@app.route("/show_similar_doc2vec_tickets/<instance_id>/<field_option>")
def show_similar_doc2vec_tickets(instance_id, field_option):
    df = read_df(model_.filename)
    try:
        idx = df[df['instanceId'] == unicode(instance_id)].index.tolist()[0]
    except:
        return 'no such instance'
    if int(field_option) == 1:
        resutls = model_.show_similar('SENT_{}'.format(idx), top_n=10, field='text')
        to_show = {}
        for idx, text in resutls.iteritems():
            if isinstance(eval_utf(text), list):
                to_show[idx] = eval_utf(text)[0]
            else:
                to_show[idx] = text
        return json.dumps(to_show, encoding='utf8', indent=2)
    else:
        to_show = model_.show_similar('SENT_{}'.format(idx), top_n=10, model_field=True)
        return json.dumps(to_show, encoding='utf8', indent=2)


if __name__ == '__main__':

    freeze_support()

    df_file_ = '../../data/LibertyGlobal/DBSCAN/config1/data_frame_selectedpreprocessed.p.zip'
    field_text_ = 'textpreprocessed'
    model_file_ = 'tmpdoc2vec.p.zip'  # ''..\..\models\LibertyGlobal\DBSCAN\config1\doc2vec.p.zip'
    model_ = CorpusToDoc2Vec(
        df_file_, field_text_, model_file_, vec_size=500, window=8, min_df=5, iter=20, workers=3
    )

    corpus_vectors = model_.get_vectors_as_np()

    app.run()


