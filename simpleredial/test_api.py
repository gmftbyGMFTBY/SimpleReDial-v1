import http.client
import torch
import random
import numpy as np
from tqdm import tqdm
import pprint
import json
import ipdb
from dataloader import *
from config import *
import argparse

def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--size', type=int, default=1000)
    parser.add_argument('--block_size', type=int, default=10)
    parser.add_argument('--topk', type=int, default=10, help='topk candidates for recall')
    parser.add_argument('--mode', type=str, default='rerank/recall/pipeline')
    parser.add_argument('--url', type=str, default='9.91.66.241')
    parser.add_argument('--port', type=int, default=22335)
    parser.add_argument('--dataset', type=str, default='douban')
    parser.add_argument('--seed', type=float, default=0.0)
    parser.add_argument('--prefix_name', type=str, default='')
    return parser.parse_args()

def load_pipeline_data(path, size=1000):
    '''for pipeline and recall test'''
    data = read_text_data_utterances(path, lang='zh')
    dataset = []
    for i in range(0, len(data), 10):
        session = data[i:i+10]
        cache = []
        for label, utterances in session:
            if label == 1:
                cache.append(utterances[-1])
        # NOTE:
        dataset.append({
            'ctx': utterances[:-1],
            'res': cache
        })
    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    collector = []
    for item in tqdm(dataset):
        collector.append({
            'segment_list': [{
                'str': item['ctx'], 
                'status': 'editing',
                'ground-truth': item['res'],
            }],
            'lang': 'zh',
            'topk': args['topk'],
        })
    print(f'[!] collect {len(collector)} samples for pipeline agent')
    return collector


def load_fake_partial_rerank_data(path, size=1000):
    # make sure the data reader
    if args['dataset'] in ['douban', 'ecommerce', 'ubuntu', 'lccc', 'lccc-large', 'restoration-200k']:
        dataset_ = read_text_data_utterances(path, lang='zh')
        # dataset = [(utterances[:-1], utterances[-1], None) for _, utterances in dataset]
        dataset = []
        for label, utterances in dataset_:
            if label == 1:
                ctx = ' '.join(utterances[:-1])
                num = random.choice([2, 3, 4])
                context = ctx[:-num]
                res = f'{ctx[-num:]} {utterances[-1]}'
                candidates = random.sample(utterances[:-1], 2)
                dataset.append((context, res, candidates))
    else:
        dataset, _ = read_json_data(path, lang='zh')
    data = []
    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    for i in dataset:
        if current_num == block_size:
            data.append({
                'segment_list': [
                    {
                        'context': j[0], 
                        'candidates': [j[1]] + j[2]
                    } for j in cache
                ],
                'lang': 'zh',
            })
            current_num, cache = 1, [i]
            block_size = random.randint(1, args['block_size'])
        else:
            current_num += 1
            cache.append(i)
    data = random.sample(data, size)
    return data


def load_fake_rerank_data(path, size=1000):
    # make sure the data reader
    dataset, _ = read_json_data(path, lang='zh')
    data = []
    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    for i in dataset:
        if current_num == block_size:
            data.append({
                'segment_list': [
                    {
                        'context': ' [SEP] '.join(j[0]), 
                        'candidates': [j[1]] + j[2]
                    } for j in cache
                ],
                'lang': 'zh',
            })
            current_num, cache = 1, [i]
            block_size = random.randint(1, args['block_size'])
        else:
            current_num += 1
            cache.append(i)
    data = random.sample(data, size)
    return data

def load_fake_recall_data(path, size=1000):
    '''for pipeline and recall test'''
    if args['dataset'] in ['douban', 'ecommerce', 'ubuntu', 'lccc', 'lccc-large', 'restoration-200k']:
        # test set only use the context 
        dataset = read_text_data_utterances(path, lang='zh')
        dataset = [(utterances[:-1], utterances[-1], None) for label, utterances in dataset]
        dataset = [dataset[i] for i in range(0, len(dataset), 10)]
    elif args['dataset'] in ['poetry', 'novel_selected']:
        dataset = read_text_data_with_source(path, lang='zh')
    else:
        dataset, _ = read_json_data(path, lang='zh')
    data = []
    cache, block_size = [], random.randint(1, args['block_size'])
    current_num = 0
    for i in tqdm(dataset):
        if current_num == block_size:
            data.append({
                'segment_list': [
                    {
                        'str': j[0], 
                        'status': 'editing'
                    } for j in cache
                ],
                'lang': 'zh',
                'topk': args['topk'],
            })
            current_num, cache = 1, [i]
            block_size = random.randint(1, args['block_size'])
        else:
            current_num += 1
            cache.append(i)
    if cache:
        data.append({
            'segment_list': [
                {
                    'str': j[0], 
                    'status': 'editing'
                } for j in cache
            ],
            'lang': 'zh',
            'topk': args['topk'],
        })
    # data = random.sample(data, size)
    print(f'[!] collect {len(data)} samples for pipeline agent')
    return data

def SendPOST(url, port, method, params):
    '''
    import http.client

    parameters:
        1. url: 9.91.66.241
        2. port: 8095
        3. method:  /rerank or /recall
        4. params: json dumps string
    '''
    headers = {"Content-type": "application/json"}
    conn = http.client.HTTPConnection(url, port)
    conn.request('POST', method, params, headers)
    response = conn.getresponse()
    code = response.status
    reason = response.reason
    data = json.loads(response.read().decode('utf-8'))
    conn.close()
    return data

def test_recall(args):
    data = load_fake_recall_data(
        f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
        size=args['size'],
    )
    # recall test begin
    avg_times = []
    collections = []
    error_counter = 0
    pbar = tqdm(data)
    for data in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/recall', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
        else:
            collections.append(rest)
            avg_times.append(rest['header']['core_time_cost_ms'])
        pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
    avg_t = round(np.mean(avg_times), 4)
    print(f'[!] avg recall time cost: {avg_t} ms; error ratio: {round(error_counter/len(data), 4)}')
    return collections

def test_partial_rerank(args):
    data = load_fake_partial_rerank_data(
        f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
        size=args['size'],
    )
    # rerank test begin
    avg_times = []
    collections = []
    error_counter = 0
    pbar = tqdm(data)
    for data in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/rerank', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
        else:
            collections.append(rest)
            avg_times.append(rest['header']['core_time_cost_ms'])
        pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
    avg_t = round(np.mean(avg_times), 4)
    print(f'[!] avg rerank time cost: {avg_t} ms; error ratio: {round(error_counter/len(data), 4)}')
    return collections


def test_rerank(args):
    data = load_fake_rerank_data(
        f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
        size=args['size'],
    )
    # rerank test begin
    avg_times = []
    collections = []
    error_counter = 0
    pbar = tqdm(data)
    for data in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/rerank', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
        else:
            collections.append(rest)
            avg_times.append(rest['header']['core_time_cost_ms'])
        pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
    avg_t = round(np.mean(avg_times), 4)
    print(f'[!] avg rerank time cost: {avg_t} ms; error ratio: {round(error_counter/len(data), 4)}')
    return collections

def test_pipeline(args):
    data = load_pipeline_data(
        f'{args["root_dir"]}/data/{args["dataset"]}/test.txt',
        size=args['size'],
    )
    # pipeline test begin
    avg_times = []
    avg_recall_times = []
    avg_rerank_times = []
    collections = []
    error_counter = 0
    pbar = tqdm(list(enumerate(data)))
    for idx, data in pbar:
        data = json.dumps(data)
        rest = SendPOST(args['url'], args['port'], '/pipeline_evaluation', data)
        # rest = SendPOST(args['url'], args['port'], '/pipeline', data)
        if rest['header']['ret_code'] == 'fail':
            error_counter += 1
            print(f'[!] ERROR happens in sample {idx}')
        else:
            collections.append(rest)
            avg_times.append(rest['header']['core_time_cost_ms'])
            avg_recall_times.append(rest['header']['recall_core_time_cost_ms'])
            avg_rerank_times.append(rest['header']['rerank_core_time_cost_ms'])
        pbar.set_description(f'[!] time: {round(np.mean(avg_times), 2)} ms; error: {error_counter}')
    # show the result
    for name in ['R@1000', 'R@500', 'R@100', 'R@50', 'MRR']:
        print(f'{name}: {rest["results"][name]}')
    avg_t = round(np.mean(avg_times), 4)
    avg_recall_t = round(np.mean(avg_recall_times), 4)
    avg_rerank_t = round(np.mean(avg_rerank_times), 4)
    print(f'[!] avg time cost: {avg_t} ms; avg recall time cost: {avg_recall_t} ms; avg rerank time cost {avg_rerank_t} ms; error ratio: {round(error_counter/len(data), 4)}')
    return collections


if __name__ == '__main__':
    # topk rewrite
    args = vars(parser_args())

    # set the random seed
    random.seed(args['seed'])

    args['root_dir'] = '/apdcephfs/share_916081/johntianlan/MyReDial'
    MAP = {
        'recall': test_recall,
        'rerank': test_rerank,
        'partial_rerank': test_partial_rerank,
        'pipeline': test_pipeline,
    }
    collections = MAP[args['mode']](args)
    
    # write into log file
    write_path = f'{args["root_dir"]}/data/{args["dataset"]}/test_api_{args["mode"]}_{args["port"]}_{args["prefix_name"]}_log.txt'
    with open(write_path, 'w') as f:
        for sample in tqdm(collections):
            data = sample['item_list']
            if sample['header']['ret_code'] == 'fail':
                continue
            if args['mode'] == 'pipeline':
                for item in data:
                    string = '\t'.join(item['context'])
                    f.write(f'[Context ] {string}\n')
                    f.write(f'[Response] {item["response"]}\n\n')
                    # f.write(f'[MRR Metric] {item["mrr"]}\n\n')
            elif args['mode'] == 'recall':
                for item in data:
                    f.write(f'[Context] {item["context"]}\n')
                    for idx, neg in enumerate(item['candidates']):
                        f.write(f'[Cands-{idx}] {neg["text"]}\n')
                    f.write('\n')
            elif args['mode'] in ['rerank', 'partial_rerank']:
                for item in data:
                    f.write(f'[Context] {item["context"]}\n')
                    for i in item['candidates']:
                        f.write(f'[Score {round(i["score"], 2)}] {i["str"]}\n')
                    f.write('\n')
            else:
                raise Exception(f'[!] Unkown mode: {args["mode"]}')

    print(f'[!] write the log into file: {write_path}')
