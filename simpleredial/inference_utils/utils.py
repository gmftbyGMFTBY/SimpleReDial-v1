from header import *
from model import *
from config import *
from dataloader import *

class Searcher:

    '''If q-q is true, the corpus is a list of tuple(context, response);
    If q-r is true, the corpus is a list of strings;
    
    Source corpus is a dict:
        key is the title, value is the url(maybe the name)
    if with_source is true, then self.if_q_q is False (only do q-r matching)'''

    def __init__(self, index_type, dimension=768, q_q=False, with_source=False, nprobe=1):
        if index_type.startswith('BHash') or index_type in ['BFlat']:
            binary = True
        else:
            binary = False
        if binary:
            self.searcher = faiss.index_binary_factory(dimension, index_type)
        else:
            self.searcher = faiss.index_factory(dimension, index_type)
        self.corpus = []
        self.binary = binary
        self.with_source = with_source
        self.source_corpus = {}
        self.if_q_q = q_q
        self.nprobe = nprobe

    def _build(self, matrix, corpus, source_corpus=None, speedup=False):
        '''dataset: a list of tuple (vector, utterance)'''
        self.corpus = corpus 
        if speedup:
            self.move_to_gpu()
        self.searcher.train(matrix)
        self.searcher.add(matrix)
        if self.with_source:
            self.source_corpus = source_corpus
        if speedup:
            self.move_to_cpu()
        print(f'[!] build collection with {self.searcher.ntotal} samples')
    
    def _search_dis(self, vector, topk=20):
        '''return the distance'''
        self.searcher.nprobe = self.nprobe
        D, I = self.searcher.search(vector, topk)
        if self.with_source:
            # pack up the source information and return
            # return the tuple (text, title, url)
            rest = [[(self.corpus[i][0], self.corpus[i][1], self.source_corpus[self.corpus[i][1]]) for i in N] for N in I]
        elif self.if_q_q:
            # the response is the second item in the tuple
            rest = [[self.corpus[i][1] for i in N] for N in I]
        else:
            rest = [[self.corpus[i] for i in N] for N in I]
            distance = [[i for i in N] for N in D]
        return rest, distance

    def _search(self, vector, topk=20):
        self.searcher.nprobe = self.nprobe
        D, I = self.searcher.search(vector, topk)
        if self.with_source:
            # pack up the source information and return
            # return the tuple (text, title, url)
            # rest = [[(self.corpus[i][0], self.corpus[i][1], self.source_corpus[self.corpus[i][1]]) for i in N] for N in I]
            rest = [[(self.corpus[i], self.source_corpus[i][0]) for i in N] for N in I]
        elif self.if_q_q:
            # the response is the second item in the tuple
            rest = [[self.corpus[i][1] for i in N] for N in I]
        else:
            rest = [[self.corpus[i] for i in N] for N in I]
        return rest

    def save(self, path_faiss, path_corpus, path_source_corpus=None):
        if self.binary:
            faiss.write_index_binary(self.searcher, path_faiss)
        else:
            faiss.write_index(self.searcher, path_faiss)
        with open(path_corpus, 'wb') as f:
            joblib.dump(self.corpus, f)
        if self.with_source:
            with open(path_source_corpus, 'wb') as f:
                joblib.dump(self.source_corpus, f)

    def load(self, path_faiss, path_corpus, path_source_corpus=None):
        if self.binary:
            self.searcher = faiss.read_index_binary(path_faiss)
        else:
            self.searcher = faiss.read_index(path_faiss)
        with open(path_corpus, 'rb') as f:
            self.corpus = joblib.load(f)
        print(f'[!] load {len(self.corpus)} utterances from {path_faiss} and {path_corpus}')
        if self.with_source:
            with open(path_source_corpus, 'rb') as f:
                self.source_corpus = joblib.load(f)

    def add(self, vectors, texts):
        '''the whole source information are added in _build'''
        self.searcher.add(vectors)
        self.corpus.extend(texts)
        print(f'[!] add {len(texts)} dataset over')

    def move_to_gpu(self, device=0):
        # self.searcher = faiss.index_cpu_to_all_gpus(self.searcher)
        res = faiss.StandardGpuResources()
        self.searcher = faiss.index_cpu_to_gpu(res, device, self.searcher)
        print(f'[!] move index to GPU device: {device} over')
    
    def move_to_cpu(self):
        self.searcher = faiss.index_gpu_to_cpu(self.searcher)
        print(f'[!] move index from GPU to CPU over')


def init_recall(args):
    searcher = Searcher(args['index_type'], dimension=args['dimension'], with_source=args['with_source'], nprobe=args['index_nprobe'])
    model_name = args['model']
    pretrained_model_name = args['pretrained_model']
    searcher.load(
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt',
        f'{args["root_dir"]}/data/{args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt',
        path_source_corpus=None,
    )
    print(f'[!] load faiss over')
    agent = load_model(args) 
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)
    print(f'[!] load model over')
    return searcher, agent


class RecallAgent:

    def __init__(self, args):
        self.searcher, self.agent = init_recall(args)
        self.args = args

    def work(self, batch, topk=None):
        '''batch: a list of string (query)'''
        batch = [i['str'] for i in batch]
        topk = topk if topk else self.args['topk']
        vectors = self.agent.encode_queries(batch)    # [B, E]
        rest_ = self.searcher._search(vectors, topk=topk)
        rest = []
        for item in rest_:
            cache = []
            for i in item:
                cache.append({'text': i})
            rest.append(cache)
        return rest

def remove_duplicate_punctuation(utterance):
    chars = []
    punctuations = ['。', '.', '！', '#', '~', '～', '?', '!', '？', '·', ' ', '）', '（', '(', ')', '{', '}']
    for i in utterance:
        if i in punctuations:
            if len(chars) > 0 and i == chars[-1]:
                continue
        chars.append(i)
    return ''.join(chars)


def load_utterances_test(args, path):
    data = read_text_data_utterances(path, args['lang'])
    data = [(label, us) for label, us in data]
    dataset = []
    for i in range(0, len(data), 10):
        batch = data[i:i+10]
        label = [i[0] for i in batch]
        sample = [i[1] for i in batch]
        ctx = sample[0][:-1]
        res = [i[-1] for i in sample]
        dataset.append((ctx, res, label))
    return dataset

def load_utterances(args, path):
    utterances = read_response_data_full(path, lang=args['lang'], turn_length=5)
    utterances = list(set(utterances))
    interval = len(utterances) // dist.get_world_size()
    chunks = []
    for i in range(0, len(utterances), interval):
        chunks.append(utterances[i:i+interval])
    chunks = chunks[:dist.get_world_size()]
    utterances = chunks[args['local_rank']]
    print(f'[!] collect {len(utterances)} for process: {args["local_rank"]}')
    return utterances

def load_agent(args):
    recall_args = load_deploy_config('recall')
    recall_args['dataset'] = args['dataset']
    recall_args['model'] = args['model']
    args.update(recall_args)
    args['tokenizer'] = args['tokenizer'][args['lang']]
    args['pretrained_model'] = args['pretrained_model'][args['lang']]
    recallagent = RecallAgent(args)
    print(f'[!] load the recall agents over')
    return recallagent

def combine_all_generate_samples(args):
    if args['local_rank'] == 0:
        dataset = []
        with open(f'{args["root_dir"]}/data/{args["dataset"]}/train_gen_ext.txt', 'w') as fw:
            for i in range(dist.get_world_size()):
                with open(f'{args["root_dir"]}/data/{args["dataset"]}/train_gen_ext_{args["local_rank"]}.txt') as f:
                    for line in f.readlines():
                        dataset.append(line)
            for line in dataset:
                fw.write(line)

def combine_all_generate_samples_pt(args):
    if args['local_rank'] == 0:
        path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_simcse.pt'
        dataset = {}
        for i in range(dist.get_world_size()):
            data = torch.load(f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_simcse_{i}.pt')
            dataset.update(data)
        print(f'[!] obtain the simcse augmentation dataset: {len(dataset)}')
        torch.save(dataset, path)
        print(f'[!] save data into {path}')

def remove_duplicate_and_hold_the_order(utterances):
    counter = set()
    data = []
    for u in utterances:
        if u in counter:
            continue
        else:
            data.append(u)
        counter.add(u)
    return data
