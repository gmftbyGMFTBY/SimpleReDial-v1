import random
import ipdb

def read(path):
    with open(path) as f:
        lines = [i.strip() for i in f.readlines()]
    datasets = []
    for i in range(0, len(lines), 10):
        datasets.append(lines[i:i+10])
    return datasets

def write(path, dataset):
    with open(path, 'w') as f:
        for item in dataset:
            for utterance in item:
                f.write(f'{utterance}\n')

def make(dataset):
    new_datasets = []
    for item in dataset:
        idx = random.randint(1, 9)
        ctx = item[0].split('\t')[1:-1]
        utterance = random.choice(ctx)
        ctx = '\t'.join(ctx)
        item[idx] = f'0\t{ctx}\t{utterance}'
        new_datasets.append(item)
    return new_datasets

if __name__ == "__main__":
    random.seed(50)
    datasets = ['ubuntu', 'douban', 'ecommerce']
    for d in datasets:
        read_path = f'{d}/test.txt'
        write_path = f'{d}/test_adv.txt'
        new_dataset = make(read(read_path))
        write(write_path, new_dataset)
