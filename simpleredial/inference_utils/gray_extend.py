from inference import *
from header import *
from dataloader.utils import *
from config import *
from .utils import *

'''
self-play strategy to generate the additional data samples for training

Make sure the deploy config is set in the config/base.yaml
'''

def gray_extend_strategy(args):
    # set the seed
    random.seed(args['seed'])

    # ext turn size
    ext_turn_size = args['ext_turn_size']

    # load the context embeddings of the extra data samples
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    dataset_ = read_text_data_utterances(path, lang=args['lang'])
    dataset = []
    for label, utterances in dataset_:
        if label == 1:
            for utterance in utterances:
                dataset.append([utterance])

    # distributed
    overall_size = len(dataset)
    interval = len(dataset) // dist.get_world_size()
    chunks = []
    for i in range(0, len(dataset), interval):
        chunks.append(dataset[i:i+interval])
    if len(chunks) > dist.get_world_size():
        c = []
        for i in chunks:
            if len(c) < dist.get_world_size():
                c.append(i)
            else:
                c[-1].extend(i)
        chunks = c
    assert len(chunks) == dist.get_world_size()
    dataset = chunks[args['local_rank']]
    print(f'[!] overall samples: {overall_size}; collect {len(dataset)} samples to extend for process: {args["local_rank"]}')

    # load agent
    recallagent = load_agent(args)

    # self-play to extend the multi turn dialog 
    batch_size = args['batch_size']
    topk = args['gen_dataset_topk']

    for _ in tqdm(range(ext_turn_size)):
        idx, invalid_num = 0, 0
        with tqdm(total=len(dataset)) as pbar:
            while idx < len(dataset):
                samples = dataset[idx:idx+batch_size]
                recall_inpt = [{'str': session} for session in samples]
                candidates = recallagent.work(recall_inpt, topk=topk)
                for index in range(idx, idx+batch_size):
                    candidate = candidates[index-idx]
                    session = samples[index-idx]
                    candidate = [remove_duplicate_punctuation(i['text']) for i in candidate]
                    candidate = [u for u in candidate if u not in session]
                    if len(candidate) == 0:
                        invalid_num += 1
                    else:
                        dataset[index].append(candidate[0])
                idx += batch_size
                pbar.update(batch_size)
                pbar.set_description(f'[!] {invalid_num} samples cannot find the appropriate response')

    # write into new file
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gen_ext_{args["local_rank"]}.txt'
    with open(path, 'w') as f:
        for item in tqdm(dataset):
            text = '\t'.join(item)
            string = f'1\t{text}\n'
            f.write(string)
    print(f'[!] generate {len(dataset)} new samples, save into {path}')
