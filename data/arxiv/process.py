import random
from tqdm import tqdm
import json
import ipdb

def load_file(path):
    with open(path) as f:
        dataset = []
        pbar = tqdm(f.readlines())
        for line in pbar:
            dataset.append(line)
    test_idx = list(range(len(dataset)))
    test_idx = random.sample(test_idx, test_size)
    test_set = [dataset[i] for i in test_idx]
    train_set = [dataset[i] for i in range(len(dataset)) if i not in test_idx]
    return test_set, train_set

def parse_data(path, data, ctx_length):
    def _split_utterances(text):
        us = [i.strip() for i in text.split('\t')]
        d = []
        end = 0
        for i in range(ctx_length, len(us), ctx_length):
            ctx, res = us[end:i], us[i]
            if ctx and res:
                d.append({'q': us[end:i], 'r': us[i]})
                end = i + 1
        return d

    with open(path, 'w') as f:
        for item in tqdm(data):
            dataset = []
            item = json.loads(item)
            dataset.extend(_split_utterances(item['abstract']))
            dataset.extend(_split_utterances(item['contents']))
            if path == 'train.txt':
                # too big, save half of the samples for training
                dataset = random.sample(dataset, int(len(dataset)/2))
            for d in dataset:
                # too huge, save half samples for training
                d = json.dumps(d)
                f.write(f'{d}\n')

if __name__ == '__main__':
    random.seed(10)

    test_size = 1000
    ctx_length = 5

    test_set, train_set = load_file('arxiv.json')
    test_set, train_set = parse_data('test.txt', test_set, ctx_length), parse_data('train.txt', train_set, ctx_length)
