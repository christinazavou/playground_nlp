from __future__ import unicode_literals

import time

from src.utils.utils import read_df, var_to_str

from src.elasticsearch import Elasticsearch
from src.elasticsearch import streaming_bulk

body_ = {
    "settings": {
        "number_of_shards": 5,
        "number_of_replicas": 1,
    }
}


class EsConnection(object):

    def __init__(self, host=None):
        if not host:
            host = {"host": "localhost", "port": 9200}
            self.con = Elasticsearch(hosts=[host])

    def create_index(self, index, shards=5, replicas=1, body=None):
        if self.con.indices.exists(index):
            to_delete = raw_input('index {} exists. want to recreate it?'.format(index))
            if to_delete != 'yes':
                return
            print "deleting {} index...".format(index)
            self.con.indices.delete(index=index)
        if not body:
            body = {
                "settings": {
                    "number_of_shards": shards,
                    "number_of_replicas": replicas
                }
            }
        print "creating {} index...".format(index)
        self.con.indices.create(index=index, body=body)
        time.sleep(50)

    def bulk_tickets(self, index_name, actions):
        for ok, result in streaming_bulk(
                self.con,
                actions=actions,
                index=index_name,
                doc_type='ticket',
                chunk_size=50  # keep the batch sizes small for appearances only
        ):
            action, result = result.popitem()
            doc_id = '/%s/ticket/%s' % (index_name, result['_id'])
            # process the information from ES whether the document has been successfully indexed
            if not ok:
                print('Failed to %s document %s: %r' % (action, doc_id, result))
                exit()
            else:
                print(doc_id)

    def update_tickets(self, index_name, body, doc_id):
        body = {"doc": {body}}
        self.con.update(index=index_name, doc_type='ticket', id=doc_id, body=body, refresh=True)


def parse_tickets(df_file, cluster_field, texts_fields, clusters_labels=None):
    df = read_df(df_file)
    for idx, row in df.iterrows():
        cluster = int(row[cluster_field])
        if clusters_labels:
            labels = clusters_labels[repr(cluster)] if repr(cluster) in clusters_labels.keys() else u''
        ticket = {
            '_id': row['instanceId'],
            '_source': {
                'timestamp': row['timestamp'],
                'cluster': cluster
            }
        }
        for field in texts_fields:
            ticket['_source'][field] = var_to_str(row[field])
        if clusters_labels:
            ticket['_source']['cluster_tokens'] = unicode(labels)
        yield ticket


def bulk_wrapper(name, body, df_file, cluster_field, text_fields):
    con = EsConnection()
    con.create_index(name, body=body)
    con.bulk_tickets(name, parse_tickets(df_file, cluster_field, text_fields))


if __name__ == '__main__':

    df_file_ = '....zip'
    df = read_df(df_file_)

    text_fields_ = ['clustering-text-subject', 'detailed_description', 'resolution']
    cluster_field_ = 'dh-cluster-description'
    name_ = 'some_data'.lower()

    bulk_wrapper(name_, body_, df_file_, cluster_field_, text_fields_)

