from config import *
from dataloader.utils import *
from .es_utils import *
import argparse
import random
import ipdb

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='douban', type=str)
    parser.add_argument('--recall_mode', default='q-r', type=str, help='q-q/q-r')
    return parser.parse_args()

def q_q_dataset(args):
    train_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    train_data = load_qa_pair(train_path, lang=args['lang'])
    print(f'[!] collect {len(train_data)} sentences for BM25 retrieval')
    return train_data

def q_r_dataset(args):
    train_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    # extend_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
    train_data = load_sentences(train_path, lang=args['lang'])
    # extend_data = load_extended_sentences(extend_path)
    # data = train_data + extend_data
    # data = list(set(data))
    data = list(set(train_data))
    print(f'[!] collect {len(data)} sentence for BM25 retrieval')
    return data

if __name__ == "__main__":
    args=  vars(parser_args())
    args['mode'] = 'test'
    args['model'] = 'dual-bert'
    config = load_config(args)
    args.update(config)
    print('test', args)

    random.seed(args['seed'])
    if args['recall_mode'] == 'q-q':
        data = q_q_dataset(args)
    else:
        data = q_r_dataset(args)
    builder = ESBuilder(
        f'{args["dataset"]}_{args["recall_mode"]}',
        create_index=True,
        q_q=True if args['recall_mode'] == 'q-q' else False,
    )

    builder.insert(data)
