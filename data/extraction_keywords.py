import nltk
import numpy as np
import torch
from tqdm import tqdm
import ipdb
from collections import Counter
import argparse
import jieba.posseg as pseg
from jieba import analyse

'''Extraction the keywords from the corpus, only extrate the verb, nous, ad wordsj'''

def read_text_data_train(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            if line[0] == '0':
                # ignore the negative samples
                continue
            line = [''.join(i.split()) for i in line[1:]]
            context, response = ' '.join(line[:-1]), line[-1]
            dataset.append((context, response))
        print(f'[!] load {len(dataset)} samples from {path}')
    return dataset

def read_text_data_test(path):
    # noted that test mode donot ignore the negative samples
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            line = [''.join(i.split()) for i in line[1:]]
            context, response = ' '.join(line[:-1]), line[-1]
            dataset.append((context, response))
        print(f'[!] load {len(dataset)} samples from {path}')
    return dataset

def extract_one_utterance_en(utterance):
    utterance = utterance.lower()
    words = [word for word, tag in nltk.pos_tag(nltk.word_tokenize(utterance)) if tag in ['VB', 'NN', 'JJ']]
    return words

def extract_one_utterance_zh(utterance):
    # jieba.analyse
    # words = [word for word, tag in pseg.cut(utterance) if tag in ['v', 'a', 'n']]
    # so slow
    words = list(analyse.extract_tags(utterance, allowPOS=('v', 'a', 'n', 'nz')))
    return words

def extract_corpus(dataset, lang='zh'):
    extract_func = extract_one_utterance_zh if lang == 'zh' else extract_one_utterance_en
    words = []
    corpus_kw = []
    for context, response in tqdm(dataset):
        ctx_kw = extract_func(context)
        res_kw = extract_func(response)
        words.extend(ctx_kw + res_kw)
        corpus_kw.append((ctx_kw, res_kw))
    words = Counter(words)
    print(f'[!] find {len(words)} keywords in the corpus')
    return words, corpus_kw

def extract_corpus_test(dataset, lang='zh'):
    extract_func = extract_one_utterance_zh if lang == 'zh' else extract_one_utterance_en
    corpus_kw = []
    for context, response in tqdm(dataset):
        ctx_kw = extract_func(context)
        res_kw = extract_func(response)
        corpus_kw.append((ctx_kw, res_kw))
    return corpus_kw
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='')
    parser.add_argument('--lang', default='en', type=str)
    parser.add_argument('--dataset', default='convai2', type=str)
    parser.add_argument('--size', default=50000, type=int)
    args = vars(parser.parse_args())

    for mode in ['train', 'test']:
        if mode == 'train':
            dataset = read_text_data_train(f'{args["dataset"]}/train.txt')
            keywords, corpus_kw = extract_corpus(dataset, lang=args['lang'])
            keywords = [i for i, j in keywords.most_common(args['size'])]
            print(f'[!] save top-{len(keywords)} keywords')
        else:
            dataset = read_text_data_test(f'{args["dataset"]}/test.txt')
            corpus_kw = extract_corpus_test(dataset, lang=args['lang'])
        
        # refine the corpus keywords
        keywords_set = set(keywords)
        new_corpus_kw = []
        stat = []
        for ctx_kw, res_kw in corpus_kw:
            ctx_kw = list(set(ctx_kw) & keywords_set)
            res_kw = list(set(res_kw) & keywords_set)
            new_corpus_kw.append((ctx_kw, res_kw))
            stat.append((len(ctx_kw), len(res_kw)))
        avg_ctx_kw = round(np.mean([i[0] for i in stat]), 4)
        avg_res_kw = round(np.mean([i[1] for i in stat]), 4)
        print(f'[{mode}] average keywords(ctx|res): {avg_ctx_kw}|{avg_res_kw}')
        torch.save((new_corpus_kw, keywords), f'{args["dataset"]}/{mode}_keywords.pt')
