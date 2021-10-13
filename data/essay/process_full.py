from tqdm import tqdm
import random
import torch

def load_file(path):
    with open(path) as f:
        dataset = []    # essay
        responses = []
        for line in f.readlines():
            line = line.strip().split('\t')
            essay_id, passage_id, _, sentence_id, _, sentence = line
            essay_id = int(essay_id)
            passage_id = int(passage_id)
            sentence_id = int(sentence_id)
            sentence = sentence.replace('|', '')
            if len(sentence) < 5:
                continue
            dataset.append((
                essay_id, passage_id, sentence_id, sentence    
            ))
            responses.append(sentence)
        responses = list(set(responses))
        print(f'[!] collect {len(dataset)} sentences')
    return dataset, responses

def write_file(dataset, path):
    with open(path, 'w') as f:
        for label, s1, s2 in dataset:
            s1 = '\t'.join(s1)
            f.write(f'{label}\t{s1}\t{s2}\n')

if __name__ == "__main__":
    random.seed(0)
    data, dataset = load_file('full.txt')
    write_file(data, 'train.txt')
