from src.utils.text_utils import add_uni_grams_and_bi_grams_from_tokens
from src.utils.text_utils import evaluate_value_or_dict
from src.utils.io_utils import read_df


def chunk_dataframe_serial(df, chunk_size):
    df_len = df.shape[0]
    for i in range(0, df_len, chunk_size):
        yield df.iloc[i:i + chunk_size]


def iter_tickets_on_field(field, input_file=None, df=None, use_bi_grams=False, as_list=True):
    """The field should be of type list (e.g. the 'textpreprocessed' field)"""
    if df is None:
        if input_file:
            df = read_df(input_file)
        else:
            raise Exception('You should give either the data frame or the input file')

    for idx, row in df.iterrows():
        text = evaluate_value_or_dict(row[field])
        yield idx, get_preprocessed_text(text, use_bi_grams, as_list)


def get_preprocessed_text(sentences_list, use_bi_grams=False, as_list=True):

    if not use_bi_grams:
        if not as_list:
            return u' '.join([item for sublist in sentences_list for item in sublist])
        else:
            return [item for sublist in sentences_list for item in sublist]

    uni_grams_and_bi_grams = set()
    for sentence in sentences_list:
        uni_grams_and_bi_grams = add_uni_grams_and_bi_grams_from_tokens(uni_grams_and_bi_grams, sentence)

    if not as_list:
        return u' '.join(uni_grams_and_bi_grams)
    else:
        return uni_grams_and_bi_grams
