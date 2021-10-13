from tqdm import tqdm
import random

def load_data_train(path):
    with open(path) as f:
        responses = []
        dataset = []
        for line in f.readlines():
            utterances = line.split('\t')
            context = utterances[:4]
            response = utterances[4]
            responses.extend(utterances[:6])
            dataset.append((context, response))
    responses = list(set(responses))
    d = []
    for c, r in tqdm(dataset):
        while True:
            neg = random.choice(responses)
            if neg != r:
                break
        d.append((c, r, neg))
    print(f'[!] collect {len(d)} samples for training')
    print(f'[!] collect {len(responses)} response utterances')
    return d, responses

def write_train_file(path, data):
    with open(path, 'w') as f:
        for c, r, neg in data:
            c = '\t'.join(c)
            f.write(f'1\t{c}\t{r}\n')
            f.write(f'0\t{c}\t{neg}\n')

if __name__ == "__main__":
    random.seed(0)
    train_set, _= load_data_train('train_.txt')
    write_train_file('train.txt', train_set)



