# from __future__ import unicode_literals
# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import json
from flask import Flask
from multiprocessing import freeze_support
from src.preprocessing.Doc2Vec import CorpusToDoc2Vec
from src.utils.pandas_utils import read_df
from src.utils.text_utils import get_variable_from_string


app = Flask(__name__)

_df_file = '../resources/exampleData.csv'
_field_text = 'textpreprocessed'
_model_file = '../resources/exampleModel.p'

model = CorpusToDoc2Vec(_df_file,
                        _field_text,
                        _model_file)

df = read_df(_df_file)


@app.route("/show_similar_doc2vec_tickets/<instance_id>/<field_option>")
def show_similar_doc2vec_tickets(instance_id, field_option):
    """
    if field_option is 1, it shows the similar documents (on the model field)
    if field option is a string, it shows that field of the similar documents
    :param instance_id:
    :param field_option:
    :return:
    """
    if int(instance_id) not in df.index:
        return 'no such instance'
    if field_option == '1':
        docs = model.show_similar_documents('SENT_{}'.format(instance_id), top_n=10, field_to_show=None)
    else:
        docs = model.show_similar_documents('SENT_{}'.format(instance_id), top_n=10, field_to_show=field_option)
    to_show = {}
    for idx, text in docs.items():
        to_show[str(idx)] = str(text)
    return json.dumps(to_show, indent=2)


if __name__ == '__main__':

    freeze_support()

    app.run(host='localhost', port=5000)
