import csv
import random
from tqdm import tqdm
import json
import ipdb
import sys
import pickle
from collections import Counter
from elasticsearch import Elasticsearch, helpers
import argparse


'''Make sure the elasticsearch has been runned for searching the hard negative samples for test set'''


def parse_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--name', default='lccc', type=str)
    parser.add_argument('--train_size', default=500000, type=int)
    parser.add_argument('--database_size', default=1000000, type=int)
    parser.add_argument('--test_size', default=10000, type=int)
    parser.add_argument('--seed', default=50, type=int)
    parser.add_argument('--mode', default='init', type=str)
    parser.add_argument('--samples', default=10, type=int)
    return parser.parse_args()


class ESUtils:

    def __init__(self, index_name, create_index=False):
        self.es = Elasticsearch(hosts=['localhost:9200'])
        self.index = index_name
        if create_index:
            mapping = {
                'properties': {
                    'response': {
                        'type': 'text',
                        'analyzer': 'ik_max_word',
                        'search_analyzer': 'ik_max_word'
                    },
                    'keyword': {
                        'type': 'keyword'
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
        for i, qa in enumerate(tqdm(pairs)):
            actions.append({
                '_index': self.index,
                '_id': i + count,
                'response': qa
            })
        helpers.bulk(self.es, actions)
        print(f'[!] database size: {self.es.count(index=self.index)["count"]}')


class ESChat:

    def __init__(self, index_name):
        self.es = Elasticsearch(hosts=['localhost:9200'])
        self.index = index_name

    def search(self, query, samples=10):
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
        hits = self.es.search(index=self.index, body=dsl, size=samples)['hits']['hits']
        rest = []
        for h in hits:
            rest.append({
                'score': h['_score'], 
                'response': h['_source']['response']
            })
        return rest


def write_file(dialogs, mode='train', samples=10):
    if mode == 'train':
        responses = [i[1] for i in dialogs]
        random.shuffle(responses)
        with open('train.txt', 'w') as f:
            for (context, response), r in tqdm(list(zip(dialogs, responses))):
                f.write(f'1\t{context}\t{response}\n')
                f.write(f'0\t{context}\t{r}\n')
    elif mode == 'test':
        chatbot = ESChat(args['name'])
        with open(f'test.txt', 'w') as f:
            error_counter = 0
            responses = [i[1] for i in dialogs]
            for context, response in tqdm(dialogs):
                rest = [i['response'] for i in chatbot.search(context, samples=samples)]
                if response in rest:
                    rest.remove(response)
                if len(rest) >= samples:
                    rest = rest[:samples-1]
                else:
                    rest.extend(random.sample(responses, samples-1-len(rest)))
                f.write(f'1\t{context}\t{response}\n')
                for i in rest:
                    f.write(f'0\t{context}\t{i}\n')


def read_file(path, mode='train'):
    with open(path, encoding='utf-8') as f:
        data = json.load(f)
        dialogs = []
        for utterances in tqdm(data):
            utterances = [''.join(i.split()) for i in utterances]
            context = '\t'.join(utterances[:-1])
            response = utterances[-1]
            dialogs.append((context, response))
    print(f'[!] load {len(dialogs)} samples')
    return dialogs


if __name__ == "__main__":
    args = vars(parse_args())
    random.seed(args['seed'])
    if args['mode'] == 'init':
        # process the train set and save into elasticsearch index
        train_data = read_file('LCCC-base_train.json', mode='train')
        train_data_ = random.sample(train_data, args['train_size'])
        write_file(train_data_, mode='train')

        esutils = ESUtils(args['name'], create_index=True)
        train_data_ = random.sample(train_data, args['database_size'])
        responses = [i[1] for i in train_data_]
        esutils.insert(responses)
    elif args['mode'] == 'retrieval':
        # build the test set
        test_data = read_file('LCCC-base_test.json', mode='test')
        test_data = random.sample(test_data, args['test_size'])
        write_file(test_data, mode='test', samples=args['samples'])
    else:
        raise Exception(f'Unknow mode: {args["mode"]}')
