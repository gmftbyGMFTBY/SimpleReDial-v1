from .es_utils import *
from tqdm import tqdm
from config import *
from dataloader.utils import *
import argparse
import json
import ipdb


'''Generate the BM25 gray candidates:
Make sure the q-q BM25 index has been built
'''


def parser_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='douban', type=str)
    parser.add_argument('--pool_size', default=1000, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--recall_mode', default='q-q', type=str)
    parser.add_argument('--topk', default=10, type=int)
    return parser.parse_args()

if __name__ == '__main__':
    args = vars(parser_args())
    bsz = args['batch_size']
    args['mode'] = 'test'
    args['model'] = 'dual-bert'    # useless
    config = load_config(args)
    args.update(config)
    args['batch_size'] = bsz

    searcher = ESSearcher(
        f'{args["dataset"]}_{args["recall_mode"]}', 
        q_q=True if args['recall_mode']=='q-q' else False
    )

    # load train dataset
    read_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    write_path = f'{args["root_dir"]}/data/{args["dataset"]}/train_bm25_gray.txt'

    # dataset = read_text_data_utterances_full(read_path, lang=args['lang'], turn_length=5)
    dataset = read_text_data_utterances(read_path, lang=args['lang'])
    data = [(utterances[:-1], utterances[-1]) for label, utterances in dataset if label == 1]
    responses = [utterances[-1] for label, utterances in dataset]
    collector = []
    pbar = tqdm(range(0, len(data), args['batch_size']))
    for idx in pbar:
        # random choice the conversation context to search the topic related responses
        context = [i[0] for i in data[idx:idx+args['batch_size']]]
        response = [i[1] for i in data[idx:idx+args['batch_size']]]
        context_str = [' '.join(i[0]) for i in data[idx:idx+args['batch_size']]]
        rest_ = searcher.msearch(context_str, topk=args['pool_size'])

        rest = []
        for gt_ctx, gt_res, i in zip(context, response, rest_):
            i = list(set(i))
            if gt_res in i:
                i.remove(gt_res)
            if len(i) < args['topk']:
                rest.append(i + random.sample(responses, args['topk']-len(i)))
            else:
                rest.append(i[:args['topk']])

        for q, r, nr in zip(context, response, rest):
            collector.append({'q': q, 'r': r, 'nr': nr})

    with open(write_path, 'w', encoding='utf-8') as f:
        for data in collector:
            string = json.dumps(data)
            f.write(f'{string}\n')
