from header import *
from model import *
from config import *
from dataloader import *
from inference import Searcher
from es.es_utils import *
from .utils import *


def init_recall(args):
    if args['model'] == 'bm25':
        # Elasticsearch
        searcher = ESSearcher(f'{args["dataset"]}_q-q', q_q=True)
        # searcher = ESSearcher(f'{args["dataset"]}_q-r', q_q=False)
        agent = None
        size = searcher.get_size()
    elif args['model'] == 'full':
        searcher = [a for _, a in load_qa_pair(f'{args["root_dir"]}/data/{args["dataset"]}/train.txt')]
        agent = None
        print(f'[!] load {len(searcher)} samples for full-rerank mode')
        size = len(searcher)
    else:
        searcher = Searcher(args['index_type'], dimension=args['dimension'], with_source=args['with_source'], nprobe=args['index_nprobe'])
        model_name = args['model']
        ipdb.set_trace()
        pretrained_model_name = args['pretrained_model'].replace('/', '_')
        if args['with_source']:
            path_source_corpus = f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_source_corpus.ckpt'
        else: 
            path_source_corpus = None
        searcher.load(
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
            f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
            path_source_corpus=path_source_corpus
        )
        searcher.move_to_gpu(device=0)
        print(f'[!] load faiss over')
        agent = load_model(args) 
        pretrained_model_name = args['pretrained_model'].replace('/', '_')
        if args['with_source']:
            save_path = f'{args["root_dir"]}/ckpt/writer/{args["model"]}/best_{pretrained_model_name}.pt'
        else:
            save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
        agent.load_model(save_path)
        print(f'[!] load model over')
        size = searcher.searcher.ntotal
    return searcher, agent, size


class RecallAgent:

    def __init__(self, args):
        self.searcher, self.agent, self.whole_size = init_recall(args)
        self.args = args

    @timethis
    def work(self, batch, topk=None):
        '''batch: a list of string (query)'''
        batch = [i['str'] for i in batch]
        topk = topk if topk else self.args['topk']
        if self.args['model'] == 'bm25':
            batch = [' '.join(i) for i in batch]
            rest_ = self.searcher.msearch(batch, topk=topk)
        elif self.args['model'] == 'full':
            rest_ = [self.searcher]
        else:
            vectors = self.agent.encode_queries(batch)    # [B, E]
            rest_ = self.searcher._search(vectors, topk=topk)
        rest = []
        for item in rest_:
            cache = []
            for i in item:
                if type(i) == str:
                    # with_source is False
                    assert self.args['with_source'] is False
                    cache.append({
                        'text': i,
                        'source': {'title': None, 'url': None},
                    })
                elif type(i) == tuple:
                    # with_source is True
                    assert self.args['with_source'] is True
                    cache.append({
                        'text': i[0],
                        'source': {
                            'title': i[1], 
                            'url': i[2],
                        }
                    })
                else:
                    raise Exception()
            rest.append(cache)
        return rest
