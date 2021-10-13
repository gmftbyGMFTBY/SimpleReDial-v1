from tqdm import tqdm
import re
import ipdb
import random
import os
import argparse


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train_size', type=int, default=1000000)
    parser.add_argument('--test_size', type=int, default=1000)
    parser.add_argument('--min_length', type=int, default=16)
    parser.add_argument('--seed', type=float, default=0.0)
    return parser.parse_args()


def load(path):
    dataset = []
    with open(path, encoding='utf-8', errors='ignore') as f:
        while len(dataset) < args['train_size'] + args['test_size']:
            line = f.readline().strip()
            if len(line) < args["min_length"]:
                continue
            item = [i.strip() for i in re.split('(。|，|！|？|，)', line) if i.strip()]
            utterances = []
            for i in item:
                if i in ['。', '，', '；', '！', '？'] and len(utterances) > 0:
                    utterances[-1] += i
                else:
                    utterances.append(i)
            if len(utterances) <= 1:
                continue
            dataset.append(line)
            if len(dataset) % 10000 == 0:
                print(f'[!] lines: {len(dataset)}', end='\r')
    return dataset


def collect(index, dataset):
    counter = 0
    n_dataset = []
    for idx in index:
        try:
            n_dataset.append(dataset[idx])
        except Exception as error:
            counter += 1
    print(f'[!] get {len(dataset)} documents; find {counter} errors')
    return n_dataset

def write(data, path):
    with open(path, 'w') as f:
        for line in data:
            f.write(f"{line}\n")
    print(f'[!] write {len(data)} samples into {path}')


if __name__ == "__main__":
    args = vars(parser_args())
    random.seed(args['seed'])
    dataset = load('/home/johntianlan/generation_data/train.txt07')
    length = len(dataset)
    print(f'[!] find {length} samples in the file')
    train_idx = random.sample(range(length), args['train_size'])
    test_idx = list(set(range(length)) - set(train_idx))
    # collect
    train_dataset = collect(train_idx, dataset)
    test_dataset = collect(test_idx, dataset)

    write(train_dataset, 'train.txt')
    write(test_dataset, 'test.txt')
