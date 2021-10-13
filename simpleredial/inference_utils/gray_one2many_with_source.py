from inference import *
from header import *
from .utils import *

'''
gray strategy generates the hard negative samples (gray samples) for each conversation context in the training and testing dataset:

Need the BERTDualInferenceFullContextDataset'''


def gray_one2many_with_source_strategy(args):
    # collect the gray negative dataset
    embds, contexts, responses = [], [], []
    for i in tqdm(range(args['nums'])):
        for idx in range(100):
            try:
                embd, context, response = torch.load(
                    f'{args["root_dir"]}/data/{args["dataset"]}/inference_context_{args["model"]}_{i}_{idx}.pt'        
                )
                embds.append(embd)
                contexts.extend(context)
                responses.extend(response)
            except:
                break
    embds = np.concatenate(embds) 
    print(f'[!] load {len(contexts)} contexts for generating the gray candidates')

    # read faiss index
    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher = Searcher(args['index_type'], dimension=args['dimension'], with_source=True, nprobe=args['index_nprobe'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_with_source_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_with_source_corpus.ckpt',
        path_source_corpus=f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_with_source_source_corpus.ckpt',
    )
    # speed up with gpu
    searcher.move_to_gpu(device=args['local_rank'])

    # search
    collection = {}
    lossing = 0
    pbar = tqdm(range(0, len(embds), args['batch_size']))
    for i in pbar:
        batch = embds[i:i+args['batch_size']]    # [B, E]
        context = contexts[i:i+args['batch_size']]
        response = responses[i:i+args['batch_size']]
        result = searcher._search(batch, topk=args['pool_size'])
        retrieved_response_pool = set()
        for c, r, rest in zip(context, response, result):
            for rest_res, rest_ctx in rest:
                if rest_ctx != c:
                    if rest_res in retrieved_response_pool:
                        collection[rest_res].append(rest_ctx)
                    else:
                        collection[rest_res] = [c, rest_ctx]
                    retrieved_response_pool.add(rest_res)
        pbar.set_description(f'[!] found {len(collection)} new samples')
    l = [len(collection[i]) for i in collection]
    print(f'[!] max contexts num: {max(l)}')
    print(f'[!] min contexts num: {min(l)}')
    print(f'[!] avg contexts num: {round(np.mean(l), 4)}')

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray.txt'
    with open(path, 'w') as f:
        for key, values in tqdm(collection.items()):
            ctxs = [values[0]] + random.sample(values[1:], 1)
            item = {'res': key, 'ctx': ctxs}
            string = json.dumps(item)
            f.write(f'{string}\n')

