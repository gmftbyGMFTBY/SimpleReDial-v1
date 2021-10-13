from inference import *
from header import *
from .utils import *

'''response strategy:
Read the candidate embeddings and save it into the faiss index

Prepare steps:
    1. generate the full context and response [Needs the Dataloader: BERTDualInferenceFullForOne2ManyDataset]
        ./scripts/inference_full_ctx_res.sh <dataset_name> dual-bert 0,1,2,3,4,5,6,7
    2. run the res-search-ctx strategy [Donot need the special dataloader]
        ./scripts/inference_gray_res_search_ctx.sh <dataset_name> dual-bert 0,1,2,3,4,5,6,7
'''

def res_search_ctx_strategy(args):
    # context faiss index
    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher = Searcher(args['index_type'], dimension=args['dimension'], nprobe=args['index_nprobe'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_full_ctx_res_ctx_faiss.ckpt',        
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_full_ctx_res_ctx_corpus.ckpt',        
    )
    searcher.move_to_gpu(device=args['local_rank'])

    res_embds, rtexts, ctexts = [], [], []
    already_added = []
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            try:
                res_embd, ctx_embd, ctext, rtext = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_full_ctx_res_{args["model"]}_{i}_{idx}.pt'
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_full_ctx_res_{args["model"]}_{i}_{idx}.pt')
            except:
                break
            res_embds.append(res_embd)
            rtexts.extend(rtext)
            ctexts.extend(ctext)
            already_added.append((i, idx))
        if len(res_embds) > 10000000:
            break

    # add the external dataset
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            if (i, idx) in already_added:
                continue
            try:
                res_embd, ctx_embd, ctext, rtext = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_full_ctx_res_{i}_{idx}.pt'
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_full_ctx_res_{i}_{idx}.pt')
            except:
                break
            res_embds.append(res_embd)
            rtexts.extend(rtext)
            ctexts.extend(ctext)
    print(f'[!] total context samples: {searcher.searcher.ntotal}')
    res_embds = np.concatenate(res_embds) 
    # search
    collection = []
    lossing = 0
    pbar = tqdm(range(0, len(res_embds), args['batch_size']))
    for i in pbar:
        batch = res_embds[i:i+args['batch_size']]    # [B, E]
        responses = rtexts[i:i+args['batch_size']]
        contexts = ctexts[i:i+args['batch_size']]
        # result = searcher._search(batch, topk=args['pool_size'])
        result, distance = searcher._search_dis(batch, topk=args['pool_size'])
        for c, r, rest, dis in zip(contexts, responses, result, distance):
            ipdb.set_trace()
            if c in rest:
                rest.remove(c)
            rest = rest[:args['gray_topk']]
            collection.append({'r': r, 'cs': rest})

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_response_one2many.txt'
    with open(path, 'w') as f:
        for item in tqdm(collection):
            string = json.dumps(item)
            f.write(f'{string}\n')
    
