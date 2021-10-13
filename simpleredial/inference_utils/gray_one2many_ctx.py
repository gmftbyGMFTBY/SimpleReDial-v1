from inference import *
from header import *
from .utils import *

'''response strategy:
Read the candidate embeddings and save it into the faiss index
Need the inference dataset: BERTDualInferenceFullForOne2ManyDataset
'''

def context_response_strategy(args):
    ctx_embds, ctexts = [], []
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
            ctx_embds.append(ctx_embd)
            ctexts.extend(ctext)
            already_added.append((i, idx))
        if len(ctx_embds) > 10000000:
            break
    ctx_embds = np.concatenate(ctx_embds) 
    # searcher
    ctx_searcher = Searcher(args['index_type'], dimension=args['dimension'])
    ctx_searcher._build(ctx_embds, ctexts, speedup=True)
    print(f'[!] train the response and context searcher over')

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
            ctx_searcher.add(ctx_embd, ctext)
    print(f'[!] total context samples: {ctx_searcher.searcher.ntotal}')
    model_name = args['model']
    pretrained_model_name = args['pretrained_model']
    ctx_searcher.save(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_full_ctx_res_ctx_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_full_ctx_res_ctx_corpus.ckpt',
    )
    print(f'[!] save context faiss index over')
