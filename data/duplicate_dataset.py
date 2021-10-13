import argparse
from tqdm import tqdm
import ipdb
import random
import os

def parser_args():
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--dataset', type=str, default='ecommerce')
    parser.add_argument('--duplicate', type=int, default=5)
    parser.add_argument('--seed', type=float, default=50)
    return parser.parse_args()


def load_dataset(path):
    with open(path) as f:
        dataset, responses = [], []
        lines = f.readlines()
        for idx in range(0, len(lines), 2):
            pos = lines[idx].strip().split('\t')
            neg = lines[idx+1].strip().split('\t')
            assert pos[0] == '1' and neg[0] == '0'
            dataset.append([pos, neg])
            responses.append(pos[-1])
            responses.append(neg[-1])
    print(f'[!] collect {len(dataset)} samples')
    return dataset, responses


def convert(sample):
    pos, neg = sample
    duplicates = random.sample(responses, args['duplicate']-1)
    samples = [pos, neg] + [['0'] + pos[1:-1] + [i] for i in duplicates]
    return samples


if __name__ == '__main__':
    args = vars(parser_args())
    random.seed(args['seed'])
    path = f'{args["dataset"]}/train.txt'
    dataset, responses = load_dataset(path)
    new_dataset = [convert(i) for i in tqdm(dataset)]
    with open(f'{args["dataset"]}/train_dup.txt', 'w') as f:
        for sample in tqdm(new_dataset):
            for i in sample:
                i = '\t'.join(i)
                f.write(f'{i}\n')
    print(f'[!] duplicate the dataset {args["dataset"]}')

    os.system(f'cp {args["dataset"]}/test.txt {args["dataset"]}/test_dup.txt')
