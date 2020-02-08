# from __future__ import unicode_literals

# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import os
import time
import logging
import numpy as np
from gensim.models.doc2vec import Doc2Vec
from gensim.models.doc2vec import TaggedDocument
from sklearn.metrics.pairwise import cosine_similarity
from scipy.spatial.distance import cosine, euclidean
from src.utils.io_utils import read_df
from src.utils.text_utils import evaluate_value_or_dict
from src.utils.utils import iter_tickets_on_field


class TaggedDocumentGenerator(object):
    def __init__(self, filename, field):
        self.filename = filename
        self.field = field

    def __iter__(self):
        df = read_df(self.filename)
        for idx, row in df.iterrows():
            text = evaluate_value_or_dict(row[self.field])
            tokens = [item for sublist in text for item in sublist]
            yield TaggedDocument(words=tokens, tags=['SENT_%s' % idx])


class CorpusToDoc2Vec(object):

    # note: if multicores not utilised i can use random_seed constant
    def __init__(self, filename, field, model_file, vector_size=400, window=8, min_count=5, iterations=10, workers=4, epochs=5):

        self.filename = filename
        self.field = field
        self.model_file = model_file

        self.vector_size = vector_size
        self.window = window
        self.min_word_count = min_count
        self.iterations = iterations
        self.workers = workers
        self.epochs = epochs

        logging.basicConfig(
            format='%(asctime)s : %(threadName)s : %(levelname)s : %(message)s',
            level=logging.INFO, filename=os.path.join(os.path.dirname(os.path.realpath(model_file)), 'trainD2V.log')
        )

        if os.path.isfile(self.model_file):
            self.model = Doc2Vec.load(self.model_file)
        else:
            self.train_model()

        logging.info('doc2vec shape: {}'.format(np.array(self.model.docvecs).shape))

    def train_model(self):
        s_time = time.time()

        inputs = list(TaggedDocumentGenerator(self.filename, self.field))  # list so that we finish with IO at once
        logging.info('Input initialized. Start training doc2vec.')

        self.model = Doc2Vec(inputs,
                             vector_size=self.vector_size,
                             window=self.window,
                             min_count=self.min_word_count,
                             workers=self.workers,
                             dbow_words=0,
                             max_vocab_size=200000,
                             iter=self.iterations,
                             # epochs=5,
                             dm_mean=1)  # , negative=5)
        self.model.save(self.model_file)

        logging.info('Finished doc2vec after {} minutes.'.format((time.time() - s_time) // 60))
        # If you're finished training a model (=no more updates, only querying), you can do
        # model.delete_temporary_training_data(keep_doctags_vectors=True, keep_inference=True)

    def get_vectors_as_np(self):
        return np.array(self.model.docvecs)

    def show_similar(self, label, top_n=10, **kwargs):
        similar_ones = self.model.docvecs.most_similar(label, topn=top_n)
        similar_labels = [int(sim.replace('SENT_', '')) for sim, p in similar_ones]

        if 'indices' in kwargs and kwargs['indices']:
            return similar_labels
        if 'model_field' in kwargs and kwargs['model_field']:
            similar_tickets = {}
            for i, (idx, tok) in enumerate(iter_tickets_on_field(self.field, self.filename)):
                if idx in similar_labels:
                    similar_tickets[unicode(idx)] = tok
            return similar_tickets
        if 'field' in kwargs:
            similar_tickets = {}
            for i, (idx, tok) in enumerate(iter_tickets_on_field(kwargs['field'], self.filename)):
                if idx in similar_labels:
                    similar_tickets[unicode(idx)] = tok
            return similar_tickets

    def similarity(self, idx1=None, idx2=None, sent1=None, sent2=None, metric='cosine'):
        if idx1 is not None and idx2 is not None:
            if metric == 'cosine':
                return 1.0 - cosine(self.model.docvecs[idx1], self.model.docvecs[idx2])
            else:  # metric == 'euclidean':
                return euclidean(self.model.docvecs[idx1], self.model.docvecs[idx2])
        elif sent1 is not None and sent2 is not None:
            if metric == 'cosine':
                return cosine_similarity(self.model.infer_vector(sent1), self.model.infer_vector(sent2))
            else:  # metric == 'euclidean'
                return euclidean(self.model.infer_vector(sent1), self.model.infer_vector(sent2))
        else:
            raise Exception('missing data to calculate similarity')

    def predict(self, doc_words):
        return self.model.infer_vector(doc_words)

    # docvec = model.model.docvecs['SENT_24327']
    # similar_by_vector(vector, topn=10, restrict_vocab=None)
    # similarity(d1, d2)
    # similarity_unseen_docs(model, doc_words1, doc_words2, alpha=0.1, min_alpha=0.0001, steps=5)
    # model_.show_similar(label='SENT_78785')
    # print model_.similarity(idx1=5791, idx2=5867), model_.similarity(idx1=5791, idx2=5867, metric='euclidean')


if __name__ == '__main__':

    df_file_ = '../../data/LibertyGlobal/DBSCAN/config1/data_frame_selectedpreprocessed.p.zip'
    field_text_ = 'textpreprocessed'
    model_file_ = 'tmpdoc2vec.p.zip'  # ''..\..\models\LibertyGlobal\DBSCAN\config1\doc2vec.p.zip'
    model_ = CorpusToDoc2Vec(
        df_file_, field_text_, model_file_, vector_size=500, window=8, min_count=5, iter=20, workers=3
    )

    corpus_vectors = model_.get_vectors_as_np()


