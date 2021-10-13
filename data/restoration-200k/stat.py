import numpy as np
import ipdb
from itertools import chain

def load(path, train=False):
    with open(path) as f:
        lines = [line.strip() for line in f.readlines() if line.strip()]
    if train:
        dataset = []
        for i in range(0, len(lines), 2):
            dataset.append(lines[i:i+2])
    else:
        dataset = []
        for i in range(0, len(lines), 10):
            dataset.append(lines[i:i+10])
    return dataset


if __name__ == "__main__":
    train_data, val_data, test_data = load('train.txt', train=True), load('valid.txt'), load('test.txt')
    print(f'[!] Size-train: {len(train_data)}')
    print(f'[!] Size-val: {len(val_data)}')
    print(f'[!] Size-test: {len(test_data)}')

    scores = [[int(j[0]) for j in i] for i in test_data]
    scores = list(chain(*scores))
    print(f'[!] postive: {scores.count(1)}')
    print(f'[!] negative: {scores.count(0)}')
    print(f'[!] ratio: {scores.count(1)/scores.count(0)}')
