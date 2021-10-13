from inference import *
from header import *
from dataloader.utils import *
from config import *
from .utils import *

'''
self-play strategy to generate the additional data samples for training

Make sure the deploy config is set in the config/base.yaml
'''

def gray_test_strategy(args):
    # set the seed
    random.seed(args['seed'])

    # load the context embeddings of the extra data samples
    path = f'{args["root_dir"]}/data/{args["dataset"]}/test.txt'
    data = load_utterances_test(args, path)
    recallagent = load_agent(args)

    # self-play
    dataset = []
    topk = args['gray_topk']
    batch_size = args['batch_size']
    for i in tqdm(range(0, len(data), batch_size)):
        batch = data[i:i+batch_size]
        recall_inpt = [{'str': ctx} for ctx, res, _ in batch]
        candidates = recallagent.work(recall_inpt, topk=topk)
        contexts = [c for c, r, l in batch]
        responses = [r for c, r, l in batch]
        labels = [l for c, r, l in batch]
        for c, r, l, cand in zip(contexts, responses, labels, candidates):
            cand = [i['text'] for i in cand]
            if r in cand:
                cand.remove(r)
            cand = cand[:5]
            dataset.append({'q': c, 'r': r, 'cand': cand, 'label': l})

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/test_gray_base.txt'
    with open(path, 'w') as f:
        for item in tqdm(dataset):
            string = json.dumps(item)
            f.write(f'{string}\n')
