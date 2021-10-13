import random
import ipdb
from tqdm import tqdm
import os
import json

# set the seed
random.seed(50)

with open('train.txt') as f, open('test.txt') as ft, open('inference.txt', 'w') as fw:
    responses = set()
    for line in tqdm(f.readlines()):
        line = json.loads(line.strip())
        utterances = line['q'] + [line['r']]
        responses |= set(utterances)
    for line in tqdm(ft.readlines()):
        line = json.loads(line.strip())
        utterances = line['q'] + [line['r']]
        responses |= set(utterances)
    responses = list(responses)
    responses = [i.strip() for i in responses if i.strip()]
    print(f'[!] collect {len(responses)} utterances')

    for response in tqdm(responses):
        fw.write(f'{response}\n')

