from header import *
from inference import *

'''
writer_with_source strategy save the writer faiss index with the source information
'''

def writer_with_source_strategy(args):
    embds, texts = [], []
    already_added = []
    source = {}
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            try:
                embd, text = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt'
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{args["model"]}_{i}_{idx}.pt')
            except:
                break
            embds.append(embd)
            texts.extend(text)
            already_added.append((i, idx))
        if len(embds) > 10000000:
            break
    for i in tqdm(range(args['nums'])):
        subsource = torch.load(f'{args["root_dir"]}/data/{args["dataset"]}/inference_subsource_{args["model"]}_{i}.pt')
        source.update(subsource)
    print(f'[!] collect {len(source)} source (title, url) pairs')
    embds = np.concatenate(embds) 
    searcher = Searcher(args['index_type'], dimension=args['dimension'], with_source=True)
    searcher._build(embds, texts, source_corpus=source)
    print(f'[!] train the searcher over')

    # add the external dataset
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            if (i, idx) in already_added:
                continue
            try:
                embd, text = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_{i}_{idx}.pt'
                )
                print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_{i}_{idx}.pt')
            except:
                break
            searcher.add(embd, text)
    print(f'[!] total samples: {searcher.searcher.ntotal}')

    model_name = args['model']
    pretrained_model_name = args['pretrained_model']
    searcher.save(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
        path_source_corpus=f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_source_corpus.ckpt',
    )
