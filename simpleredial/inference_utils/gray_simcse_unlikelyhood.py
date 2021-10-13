from inference import *
from header import *
from .utils import *


def gray_simcse_unlikelyhood_strategy(args):
    # read the embeddings
    embds, contexts, responses, indexes = [], [], [], []
    for idx in range(100):
        try:
            embd, text, res, index = torch.load(
                f'{args["root_dir"]}/data/{args["dataset"]}/inference_simcse_ctx_{args["model"]}_{args["local_rank"]}_{idx}.pt'
            )
            print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_simcse_ctx_{args["model"]}_{args["local_rank"]}_{idx}.pt')
        except:
            break
        embds.append(embd)
        contexts.extend(text)
        responses.extend(res)
        indexes.extend(index)
    embds = np.concatenate(embds)
    # read faiss index
    model_name = args['model']
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    searcher = Searcher(args['index_type'], dimension=args['dimension'], nprobe=args['index_nprobe'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
    )
    # speed up with gpu
    searcher.move_to_gpu(device=args['local_rank'])

    # search
    collection = {}
    pbar = tqdm(range(0, len(embds), args['batch_size']))
    for i in pbar:
        batch = embds[i:i+args['batch_size']]    # [B, E]
        text = contexts[i:i+args['batch_size']]
        res = responses[i:i+args['batch_size']]
        index = indexes[i:i+args['batch_size']]
        result = searcher._search(batch, topk=args['pool_size'])
        for t, pos, idx, rest in zip(text, res, index, result):
            if pos in rest:
                rest.remove(pos)
            rest = rest[:args['gray_topk']]
            if idx in collection:
                collection[idx].append({
                    'context': t,
                    'pos_response': pos,
                    'neg_responses': rest
                })
            else:
                collection[idx] = [{'context': t, 'pos_response': pos, 'neg_responses': rest}]

    # write into new file
    data = []
    for _, values in collection.items():
        data.extend(values)
    print(f'[!] total test samples: {len(data)}')
    path = f'{args["root_dir"]}/data/{args["dataset"]}/test_gray_simcse.pt'
    torch.save(data, path)
