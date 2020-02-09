# from __future__ import unicode_literals
# import warnings
# warnings.filterwarnings(action='ignore', category=UserWarning, module='gensim')

import json
from flask import Flask
from multiprocessing import freeze_support
from src.preprocessing.Doc2Vec import CorpusToDoc2Vec
from src.utils.pandas_utils import read_df
from src.utils.text_utils import evaluate_value_or_dict


app = Flask(__name__)

_df_file = '/home/christina/Documents/playground_nlp/test/exampleData.csv'
_field_text = 'textpreprocessed'
_model_file = '/home/christina/Documents/playground_nlp/test/preprocessing/exampleModel.p'

model = CorpusToDoc2Vec(_df_file,
                        _field_text,
                        _model_file)

df = read_df(_df_file)


@app.route("/show_similar_doc2vec_tickets/<instance_id>/<field_option>")
def show_similar_doc2vec_tickets(instance_id, field_option):
    try:
        idx = df[df['instanceId'] == instance_id].index.tolist()[0]
    except:
        return 'no such instance'
    if int(field_option) == 1:
        resutls = model.show_similar('SENT_{}'.format(idx), top_n=10, field='text')
        to_show = {}
        for idx, text in resutls.iteritems():
            if isinstance(evaluate_value_or_dict(text), list):
                to_show[idx] = evaluate_value_or_dict(text)[0]
            else:
                to_show[idx] = text
        return json.dumps(to_show, encoding='utf8', indent=2)
    else:
        to_show = model.show_similar('SENT_{}'.format(idx), top_n=10, model_field=True)
        return json.dumps(to_show, encoding='utf8', indent=2)


if __name__ == '__main__':

    freeze_support()

    app.run()
