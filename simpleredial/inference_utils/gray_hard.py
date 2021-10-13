from inference import *
from model import *
from header import *
from .utils import *
from es.es_utils import *


def gray_hard_strategy(args):
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
    searcher = Searcher(args['index_type'], dimension=args['dimension'], nprobe=args['index_nprobe'])
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
    )
    # speed up with gpu
    searcher.move_to_gpu(device=args['local_rank'])

    # search
    collection = []
    bad_response_num = 0
    pbar = tqdm(range(0, len(embds), args['batch_size']))
    sample_num = 0
    for i in pbar:
        batch = embds[i:i+args['batch_size']]    # [B, E]
        context = contexts[i:i+args['batch_size']]
        response = responses[i:i+args['batch_size']]
        result, distance = searcher._search_dis(batch, topk=args['gray_topk'])
        for c, r, rest_, dis_ in zip(context, response, result, distance):
            rest_ = [u for u in rest_ if u not in c]
            rest, dis = [], []
            for u, d in zip(rest_, dis_):
                if u not in c:
                    rest.append(u)
                    dis.append(d)
            distance_range = dis[-1] - dis[0]
            length = len(rest)
            if r in rest:
                r_idx = rest.index(r)
                rest.remove(r)
            else:
                r_idx = -1
            ground_range = dis[-1] - dis[r_idx]
            # 10 is the distance threshold
            if distance_range > 10 and ground_range/distance_range < 0.25:
                bad_response = True
                bad_response_num += 1
            else:
                bad_response = False
            hard_positives = rest[:10]
            collection.append({'q': c, 'r': r, 'hp': hard_positives, 'bad_response': bad_response})
        sample_num += len(batch)
        pbar.set_description(f'[!] total response: {sample_num}; bad response: {bad_response_num}')
    print(f'[!] total samples: {len(embds)}; bad response num: {bad_response_num}')

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray.txt'
    with open(path, 'w') as f:
        for item in tqdm(collection):
            string = json.dumps(item)
            f.write(f'{string}\n')

