import random
import ipdb
from tqdm import tqdm
import os
import json

# set the seed
random.seed(50)

# load
with open('train_full.txt') as f, open('train.txt', 'w', encoding='utf-8') as fw:
    dataset = []
    responses = []
    for line in tqdm(f.readlines()):
        dataset.append(line)
        line = json.loads(line)
        responses.extend([i.strip() for i in line['q'] if i.strip()])
    dataset = random.sample(dataset, 500000)
    for line in dataset:
        # replace the null string
        line = json.loads(line)
        nr = [i.strip() for i in line['nr'] if i.strip()]
        if len(nr) < 10:
            nr.extend(random.sample(responses, 10-len(nr)))
        line['nr'] = nr
        line = json.dumps(line)
        fw.write(f'{line}\n')

with open('test_full.txt') as f, open('test.txt', 'w', encoding='utf-8') as fw:
    dataset = []
    for line in tqdm(f.readlines()):
        dataset.append(line)
    dataset = random.sample(dataset, 10000)
    for line in dataset:
        # replace the null string
        line = json.loads(line)
        nr = [i.strip() for i in line['nr'] if i.strip()]
        if len(nr) < 10:
            nr.extend(random.sample(responses, 10-len(nr)))
        line['nr'] = nr
        line = json.dumps(line)
        fw.write(f'{line}\n')
