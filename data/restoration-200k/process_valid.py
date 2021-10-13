import random
import json
from tqdm import tqdm
import ipdb
from collections import Counter
from elasticsearch import Elasticsearch, helpers
import argparse
from process_train import *

class ESBuilder:

    def __init__(self, index_name, create_index=False, q_q=False):
        self.es = Elasticsearch(hosts=['localhost:9200'])
        self.index = index_name
        self.q_q = q_q

        if create_index:
            if q_q is False:
                mapping = {
                    'properties': {
                        'response': {
                            'type': 'text',
                            'analyzer': 'ik_max_word',
                            'search_analyzer': 'ik_max_word',
                        },
                        'keyword': {
                            'type': 'keyword'
                        }
                    }
                }
            else:
                mapping = {
                    'properties': {
                        'context': {
                            'type': 'text',
                            'analyzer': 'ik_max_word',
                            'search_analyzer': 'ik_max_word',
                        },
                        'response': {
                            'type': 'keyword',
                        }
                    }
                }
            if self.es.indices.exists(index=self.index):
                self.es.indices.delete(index=self.index)
            rest = self.es.indices.create(index=self.index)
            rest = self.es.indices.put_mapping(body=mapping, index=self.index)

    def insert(self, pairs):
        count = self.es.count(index=self.index)['count']
        actions = []
        for i, (q, a) in enumerate(tqdm(pairs)):
            if self.q_q:
                actions.append({
                    '_index': self.index,
                    '_id': i + count,
                    'context': q,
                    'response': a,
                })
            else:
                actions.append({
                    '_index': self.index,
                    '_id': i + count,
                    'response': a,
                    'keyword': a,
                })
        helpers.bulk(self.es, actions)
        print(f'[!] database size: {self.es.count(index=self.index)["count"]}')


class ESSearcher:

    def __init__(self, index_name, q_q=False):
        self.es = Elasticsearch(hosts=['localhost:9200'])
        self.index = index_name
        self.q_q = q_q

    def msearch(self, queries, topk=10, limit=128):
        # limit the queries length
        queries = [query[-limit:] for query in queries]

        search_arr = []
        for query in queries:
            search_arr.append({'index': self.index})
            if self.q_q:
                search_arr.append({
                    'query': {
                        'match': {
                            'context': query
                        }
                    },
                    'collapse': {
                        'field': 'response'    
                    },
                    'size': topk,
                })
            else:
                search_arr.append({
                    'query': {
                        'match': {
                            'response': query
                        }
                    },
                    'collapse': {
                        'field': 'keyword'    
                    },
                    'size': topk,
                })

        # prepare for searching
        request = ''
        for each in search_arr:
            request += f'{json.dumps(each)} \n'
        rest = self.es.msearch(body=request)

        results = []
        for each in rest['responses']:
            p = []
            try:
                for utterance in each['hits']['hits']:
                    p.append(utterance['fields']['response'][0])
            except:
                ipdb.set_trace()
            results.append(p)
        return results

    def search(self, query, topk=10):
        if self.q_q:
            dsl = {
                'query': {
                    'match': {
                        'context': query
                    }
                },
                'collapse': {
                    'field': 'response'
                }
            }
        else:
            dsl = {
                'query': {
                    'match': {
                        'response': query
                    }
                },
                'collapse': {
                    'field': 'keyword'
                }
            }
        hits = self.es.search(index=self.index, body=dsl, size=topk)['hits']['hits']
        rest = []
        for h in hits:
            rest.append(h['_source']['response'])
        return rest


def expand_train_data(train_data):
    n_train_data = []
    for session in train_data:
        ctx, res, _ = session
        ctx.append(res)
        for i in range(1, len(ctx)):
            ctx_ = ctx[:i]
            res_ = ctx[i]
            n_train_data.append((ctx_, res_))
    print(f'[!] obtain expand train dataset size: {len(n_train_data)}')
    return n_train_data

if __name__ == "__main__":
    random.seed(0)
    # process the train set and save into elasticsearch index
    train_data, _ = load_data_train('train_.txt')
    train_data = expand_train_data(train_data)
    val_data, _ = load_data_train('valid_.txt')

    esutils = ESBuilder('restoration-200k', create_index=True, q_q=True)
    data = [(' '.join(pair[0]), pair[1]) for pair in train_data + val_data]
    esutils.insert(data)
    eschat = ESSearcher('restoration-200k', q_q=True)
    inner_bsz = 32
    with open('valid.txt', 'w') as f:
        for i in range(0, len(val_data), inner_bsz):
            batch = val_data[i:i+inner_bsz]
            queries = [' '.join(i[0]) for i in batch] 
            rest = eschat.msearch(queries, topk=10, limit=128)
            responses = [i[1] for i in batch]
            context = [i[0] for i in batch]
            for c, r, cands in zip(context, responses, rest):
                if r in cands:
                    cands.remove(r)
                cands = cands[:9]
                c = '\t'.join(c)
                f.write(f'1\t{c}\t{r}\n')
                for cand in cands:
                    f.write(f'0\t{c}\t{cand}\n')
