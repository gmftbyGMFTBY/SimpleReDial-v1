from inference import *
from header import *
from .utils import *


def data_filter_strategy(args):
    ctext, rtext, scores = [], [], []
    dataset = []
    for i in tqdm(range(args['nums'])):
        c, r, s = torch.load(
            f'{args["root_dir"]}/data/{args["dataset"]}/inference_full_filter_{args["model"]}_{i}.pt'
        )
        print(f'[!] load {args["root_dir"]}/data/{args["dataset"]}/inference_full_filter_{args["model"]}_{i}.pt')
        p = [(i, j, k) for i, j, k in zip(c, r, s)]
        dataset.extend(p)

    # sort
    dataset = sorted(dataset, key=lambda x: x[2], reverse=True)
    dataset = dataset[:args['data_filter_size']]
    path = f'{args["root_dir"]}/data/{args["dataset"]}/train_full_filter_{args["data_filter_size"]}.pt'
    torch.save(dataset, path)
