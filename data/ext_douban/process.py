from tqdm import tqdm
import random
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=2000000)
    parser.add_argument('--seed', type=float, default=0.)
    parser.add_argument('--whole_size', type=int, default=107317803)
    return parser.parse_args()

def load(path):
    with open(path, encoding='gbk') as f:
        # this may take a long time
        lines = f.readlines()

        random_idx = random.sample(range(args['whole_size']), args['size'])
        lines = [lines[i] for i in random_idx]
        dataset = []
        for line in tqdm(lines):
            line = line.strip().split('\t')
            dataset.extend([line[1], line[2]])
    dataset = list(set(dataset))
    print(f'[!] collected {len(dataset)} utterances')
    return dataset

def write(dataset, path):
    with open(path, 'w') as f:
        for utterance in tqdm(dataset):
            f.write(f'{utterance}\n')

if __name__ == "__main__":
    args = vars(parser_args())
    random.seed(args['seed'])
    utterances = load('douban_gbk.txt')
    write(utterances, 'train.txt')
