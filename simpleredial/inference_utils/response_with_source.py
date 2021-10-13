from inference import *
from header import *
from .utils import *

'''response strategy:
Read the candidate embeddings and save it into the faiss index
'''

def response_with_source_strategy(args):
    embds, texts, sources = [], [], []
    already_added = []
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            try:
                embd, text, source = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_with_source_{i}_{idx}.pt'
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_with_source_{i}_{idx}.pt')
            except:
                break
            embds.append(embd)
            texts.extend(text)
            sources.extend(source)
            already_added.append((i, idx))
        if len(embds) > 10000000:
            break
    embds = np.concatenate(embds) 
    searcher = Searcher(args['index_type'], dimension=args['dimension'], with_source=True)
    searcher._build(embds, texts, sources, speedup=True)
    print(f'[!] train the searcher over')

    # add the external dataset
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            if (i, idx) in already_added:
                continue
            try:
                embd, text, source = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_with_source_{i}_{idx}.pt'
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_with_source_{i}_{idx}.pt')
            except:
                break
            searcher.add(embd, text)
    print(f'[!] total samples: {searcher.searcher.ntotal}')

    model_name = args['model']
    pretrained_model_name = args['pretrained_model']
    searcher.save(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_with_source_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_with_source_corpus.ckpt',
        path_source_corpus=f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_with_source_source_corpus.ckpt',
    )
    print(f'[!] save faiss index over')
