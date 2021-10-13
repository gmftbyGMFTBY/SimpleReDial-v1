from inference import *
from header import *
from dataloader.utils import *
from config import *
from .utils import *

'''
self-play strategy to generate the additional data samples for training

Make sure the deploy config is set in the config/base.yaml
'''

def self_play_strategy(args):
    # set the seed
    random.seed(args['seed'])

    # load the context embeddings of the extra data samples
    path = f'{args["root_dir"]}/data/{args["dataset"]}/test.txt'
    data = load_utterances_test(args, path)
    recallagent = load_agent(args)

    dataset, idx = [], 0
    batch_size = args['batch_size']
    for i in range(len(data)):
        batch = data[i:i+batch_size]
        recall_inpt = [{'str': ctx} for ctx, res in batch]
        candidates = recallagent.work(recall_inpt, topk=args['gen_dataset_topk'])
        responses = [res for _, res in batch]
        contexts = [ctx for ctx, _ in batch]
        for ctx, res, cand in zip(contexts, responses, candidates):
            if res in cand:
                cand.remove(res)
            cand = cand[:10]
            dataset.append({'q': ctx, 'r': res, 'cand': cand})
            ipdb.set_trace()

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/test_gray.txt'
    with open(path, 'w') as f:
        for item in tqdm(dataset):
            string = json.dumps(item)
            f.write(f'{string}\n')
    print(f'[!] self-play generate {len(dataset)} samples, save into {path}')
