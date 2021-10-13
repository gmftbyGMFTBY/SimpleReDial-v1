from inference import *
from header import *
from .utils import *

'''response strategy:
Read the candidate embeddings and save it into the faiss index
'''

def da_strategy(args):
    contexts, responses, results = [], [], []
    for i in tqdm(range(args['nums'])):
        c, r, re = torch.load(
            f'{args["root_dir"]}/data/{args["dataset"]}/inference_bert_mask_da_{i}.pt',
            map_location=torch.device('cpu')
        )
        print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_bert_mask_da_{i}.pt')
        contexts.extend(c)
        responses.extend(r)
        results.extend(re)
    print(f'[!] collect {len(contexts)} samples')
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_bert_mask_da_results.pt'

    # full split
    n_ctx, n_res, n_ret = [], [], []
    for c, r, re in zip(contexts, responses, results):
        utterances = c + [r]
        start_num = max(1, len(utterances) - args['full_turn_length'])
        for i in range(start_num, len(utterances)):
            n_ctx.append(utterances[:i])
            n_res.append(utterances[i])
            n_ret.append(list(set(chain(*re[:i]))))
    torch.save([n_ctx, n_res, n_ret], path)
    print(f'[!] save the data into {path}')

