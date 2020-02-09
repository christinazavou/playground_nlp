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
from src.utils.pandas_utils import iter_tickets_on_field


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
        return np.array(self.model.docvecs.vectors_docs)

    def show_similar(self, label, top_n=10, **kwargs):
        similar_documents = self.model.docvecs.most_similar(label, topn=top_n)
        labels_of_similar_documents = [int(sim.replace('SENT_', '')) for sim, p in similar_documents]

        if 'indices' in kwargs and kwargs['indices']:
            return labels_of_similar_documents
        if 'model_field' in kwargs and kwargs['model_field']:
            return self._get_similar_tickets(self.field, labels_of_similar_documents)
        if 'field' in kwargs:
            return self._get_similar_tickets(kwargs['field'], labels_of_similar_documents)

    def _get_similar_tickets(self, field_to_read, indices_of_similar_documents):
        similar_tickets = {}
        for i, (idx, tok) in enumerate(iter_tickets_on_field(field_to_read, self.filename)):
            if idx in indices_of_similar_documents:
                similar_tickets[idx] = tok
        return similar_tickets

    def similarity_given_sentences(self, sent1, sent2, metric='cosine'):
        if metric == 'cosine':
            return cosine_similarity(self.model.infer_vector(sent1), self.model.infer_vector(sent2))
        else:
            return euclidean(self.model.infer_vector(sent1), self.model.infer_vector(sent2))

    def similarity_given_indices(self, idx1, idx2, metric='cosine'):
        if metric == 'cosine':
            return 1.0 - cosine(self.model.docvecs[idx1], self.model.docvecs[idx2])
        else:
            return euclidean(self.model.docvecs[idx1], self.model.docvecs[idx2])

    def predict(self, doc_words):
        return self.model.infer_vector(doc_words)

