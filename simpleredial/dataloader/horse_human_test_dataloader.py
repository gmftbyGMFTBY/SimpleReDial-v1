from header import *
from .utils import *
from .util_func import *

'''Only for Testing'''

class HORSETestDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_horse_human_test_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        path = f'{args["root_dir"]}/data/{args["dataset"]}/horse-human-test.pkl'
        data = pickle.load(open(path, 'rb'))
        for item in tqdm(data):
            ctx = item['ctx']
            res = item['res']
            utterances = ctx + [r for r, _ in res]
            datas = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            cids, rids = datas[:len(ctx)], datas[len(ctx):]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = ids[-(self.args['max_len']-2):]
            rids = rids[:(self.args['res_max_len']-2)]
            ids = [self.cls] + ids + [self.sep]
            rids = [[self.cls] + r + [self.sep] for r in rids]
            labels = [l for _, l in res]
            assert len(labels) == len(rids)
            self.data.append({
                'label': labels,
                'ids': ids,
                'rids': rids,
                'ctext': ctx,
                'rtext': [r for r, _ in res]
            })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        rids = [torch.LongTensor(i) for i in bundle['rids']]
        return ids, rids, bundle['label'], bundle['ctext'], bundle['rtext']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, rids, label, ctext, rtext = batch[0]
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        rids_mask = generate_mask(rids)
        ids, rids, rids_mask = to_cuda(ids, rids, rids_mask)
        return {
            'ids': ids, 
            'rids': rids, 
            'rids_mask': rids_mask, 
            'label': label,
            'ctext': ctext,
            'rtext': rtext,
        }

        
class HORSETestInteractionDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_horse_human_test_interaction_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        path = f'{args["root_dir"]}/data/{args["dataset"]}/horse-human-test.pkl'
        data = pickle.load(open(path, 'rb'))
        for item in tqdm(data):
            ctx = item['ctx']
            res = item['res']
            utterances = ctx + [r for r, _ in res]
            datas = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            cids, rids = datas[:len(ctx)], datas[len(ctx):]
            ids = []
            for u in cids:
                ids.extend(u + [self.eos])
            ids.pop()
            test_ids, test_tids = [], []
            for response in rids:
                ids_ = deepcopy(ids)
                truncate_pair(ids_, response, self.args['max_len'])
                ids = [self.cls] + ids_ + [self.sep] + response + [self.sep]
                tids = [0] * (len(ids_) + 2) + [1] * (len(response) + 1)
                test_ids.append(ids)
                test_tids.append(tids)
            labels = [l for _, l in res]
            assert len(labels) == len(test_ids)
            self.data.append({
                'label': labels,
                'ids': test_ids,
                'tids': test_tids,
                'ctext': ctx,
                'rtext': [r for r, _ in res]
            })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = [torch.LongTensor(i) for i in bundle['ids']]
        tids = [torch.LongTensor(i) for i in bundle['tids']]
        return ids, tids, bundle['label'], bundle['ctext'], bundle['rtext']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, tids, label, ctext, rtext = batch[0]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, tids, ids_mask = to_cuda(ids, tids, ids_mask)
        return {
            'ids': ids, 
            'tids': tids, 
            'mask': ids_mask, 
            'label': label,
            'ctext': ctext,
            'rtext': rtext,
        }

        
class HORSETestSAInteractionDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_horse_human_sa_test_interaction_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        path = f'{args["root_dir"]}/data/{args["dataset"]}/horse-human-test.pkl'
        data = pickle.load(open(path, 'rb'))
        for item in tqdm(data):
            ctx = item['ctx']
            res = item['res']
            ids, tids, sids = [], [], []
            for r, _ in res:
                utterances = ctx + [r]
                ids_, tids_, sids_ = self.annotate(utterances)
                ids.append(ids_)
                tids.append(tids_)
                sids.append(sids_)
            labels = [l for _, l in res]
            assert len(labels) == len(ids)
            self.data.append({
                'label': labels,
                'ids': ids,
                'tids': tids,
                'sids': sids,
                'ctext': ctx,
                'rtext': [r for r, _ in res]
            })    
            
    def __len__(self):
        return len(self.data)
    
    def annotate(self, utterances):
        tokens = [self.vocab.tokenize(utt) for utt in utterances]
        ids, tids, sids, tcache, scache = [], [], [], 0, 0
        for idx, tok in enumerate(tokens[:-1]):
            ids.extend(tok)
            ids.append('[SEP]')
            tids.extend([tcache] * (len(tok) + 1))
            sids.extend([scache] * (len(tok) + 1))
            scache = 0 if scache == 1 else 1
            tcache = 0
        tcache = 1
        ids.pop()
        sids.pop()
        tids.pop()
        ids = self.vocab.convert_tokens_to_ids(ids)
        rids = self.vocab.convert_tokens_to_ids(tokens[-1])
        trids = [tcache] * len(rids)
        srids = [scache] * len(rids)
        truncate_pair_with_other_ids(ids, rids, tids, trids, sids, srids, self.args['max_len'])
        ids = [self.cls] + ids + [self.sep] + rids + [self.sep]
        tids = [0] + tids + [0] + trids + [1]
        sids = [0] + sids + [sids[-1]] + srids + [srids[-1]]
        assert len(ids) == len(tids) == len(sids)
        return ids, tids, sids

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = [torch.LongTensor(i) for i in bundle['ids']]
        tids = [torch.LongTensor(i) for i in bundle['tids']]
        sids = [torch.LongTensor(i) for i in bundle['sids']]
        return ids, tids, sids, bundle['label'], bundle['ctext'], bundle['rtext']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, tids, sids, label, ctext, rtext = batch[0]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, tids, sids, ids_mask = to_cuda(ids, tids, sids, ids_mask)
        return {
            'ids': ids, 
            'tids': tids,
            'sids': sids,
            'mask': ids_mask,
            'ctext': ctext,
            'rtext': rtext,
            'label': label
        }

        
class HORSESATestDataset(Dataset):

    '''speaker-aware and turn-aware'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_horse_human_sa_tl_test_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        path = f'{args["root_dir"]}/data/{args["dataset"]}/horse-human-test.pkl'
        data = pickle.load(open(path, 'rb'))
        for item in tqdm(data):
            ctx = item['ctx']
            res = item['res']
            utterances = ctx + [r for r, _ in res]
            datas = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            cids, rids = datas[:len(ctx)], datas[len(ctx):]
            scache, sids, ids = 0, [], []
            turn_counter, tlids = 0, []
            for u in cids:
                ids.extend(u + [self.sep])
                sids.extend([scache] * (len(u) + 1))
                scache = 0 if scache == 1 else 1
                tlids.extend([turn_counter] * (len(u) + 1))
                turn_counter += 1
            ids.pop()
            sids.pop()
            tlids.pop()
            ids = ids[-(self.args['max_len']-2):]
            sids = sids[-(self.args['max_len']-2):]
            tlids = tlids[-(self.args['max_len']-2):]
            rids = rids[:(self.args['res_max_len']-2)]
            ids = [self.cls] + ids + [self.sep]
            sids = [0] + sids + [sids[-1]]
            tlids = [0] + tlids + [tlids[-1]]
            rids = [[self.cls] + r + [self.sep] for r in rids]
            labels = [l for _, l in res]
            assert len(labels) == len(rids)
            self.data.append({
                'label': labels,
                'ids': ids,
                'sids': sids,
                'tlids': tlids,
                'rids': rids,
                'ctext': ctx,
                'rtext': [r for r, _ in res]
            })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        sids = torch.LongTensor(bundle['sids'])
        tlids = torch.LongTensor(bundle['tlids'])
        rids = [torch.LongTensor(i) for i in bundle['rids']]
        return ids, sids, tlids, rids, bundle['label'], bundle['ctext'], bundle['rtext']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, sids, tlids, rids, label, ctext, rtext = batch[0]
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        rids_mask = generate_mask(rids)
        ids, sids, tlids, rids, rids_mask = to_cuda(ids, sids, tlids, rids, rids_mask)
        return {
            'ids': ids, 
            'sids': sids, 
            'tlids': tlids, 
            'rids': rids, 
            'rids_mask': rids_mask, 
            'label': label,
            'ctext': ctext,
            'rtext': rtext,
        }

        
class HORSECompTestDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_horse_comp_human_test_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        path = f'{args["root_dir"]}/data/{args["dataset"]}/horse-human-test.pkl'
        data = pickle.load(open(path, 'rb'))
        for item in tqdm(data):
            self.data.append({
                'label': [l for _, l in item['res']],
                'context': item['ctx'],
                'responses': [r for r, _ in item['res']]
            })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        assert len(batch) == 1
        ctx, res, label = batch[0]
        return {
            'context': ctx,
            'responses': res,
            'label': label,
        }
