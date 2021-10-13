from header import *
from .utils import *
from .util_func import *

'''Only for Testing'''

class FineGrainedTestDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_fg_test_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        for fix in ['brandenwang', 'lt', 'lt2']:
            path = f'{args["root_dir"]}/data/{args["dataset"]}/fg-{fix}-test.txt'
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 7)):
                batch = data[i:i+7]
                rids = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': ['\t'.join(b[1]) for b in batch],
                    'owner': fix,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        rids = [torch.LongTensor(i) for i in bundle['rids']]
        return ids, rids, bundle['label'], bundle['text'], bundle['owner']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, rids, label, text, owner = batch[0]
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        rids_mask = generate_mask(rids)
        label = torch.LongTensor(label)
        ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
        return {
            'ids': ids, 
            'rids': rids, 
            'rids_mask': rids_mask, 
            'label': label,
            'text': text,
            'owner': owner,
        }

        
class FineGrainedTestPositionWeightDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.special_tokens = set([self.unk, self.cls, self.sep])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_fg_test_pw_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        for fix in ['brandenwang', 'lt', 'lt2']:
            path = f'{args["root_dir"]}/data/{args["dataset"]}/fg-{fix}-test.txt'
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 7)):
                batch = data[i:i+7]
                rids = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    position_w, w = [], self.args['min_w']
                    for u in cids:
                        ids.extend(u + [self.sep])
                        for token in u + [self.sep]:
                            if token not in self.special_tokens:
                                position_w.append(w)
                            else:
                                position_w.append(self.args['w_sp_token'])
                        w += self.args['w_delta']
                    ids.pop()
                    position_w.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    position_w = position_w[-(self.args['max_len']-2):]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    position_w = [w-self.args['w_delta']] + position_w + [self.args['w_sp_token']]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': ['\t'.join(b[1]) for b in batch],
                    'position_w': position_w,
                    'owner': fix,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        rids = [torch.LongTensor(i) for i in bundle['rids']]
        position_w = torch.tensor(bundle['position_w'])
        return ids, rids, position_w, bundle['label'], bundle['text'], bundle['owner']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, rids, pos_w, label, text, owner = batch[0]
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        rids_mask = generate_mask(rids)
        label = torch.LongTensor(label)
        ids, rids, pos_w, rids_mask, label = to_cuda(ids, rids, pos_w, rids_mask, label)
        return {
            'ids': ids, 
            'rids': rids, 
            'rids_mask': rids_mask, 
            'pos_w': pos_w,
            'label': label,
            'text': text,
            'owner': owner,
        }

        
class FineGrainedTestInteractionDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_fg_interaction_test_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        self.data = []
        for fix in ['brandenwang', 'lt', 'lt2']:
            path = f'{args["root_dir"]}/data/{args["dataset"]}/fg-{fix}-test.txt'
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 7)):
                batch = data[i:i+7]
                rids = []
                ids, tids = [], []
                context, responses = [], []
                for _, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids = []
                    for u in item[:-1]:
                        cids.extend(u + [self.eos])
                    cids.pop()
                    rids = item[-1]
                    truncate_pair(cids, rids, self.args['max_len'])
                    ids_ = [self.cls] + cids + [self.sep] + rids + [self.sep]
                    tids_ = [0] * (len(cids) + 2) + [1] * (len(rids) + 1)
                    ids.append(ids_)
                    tids.append(tids_)
                    responses.append(utterances[-1])
                context = ' [SEP] '.join(utterances[:-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'tids': tids,
                    'context': context,
                    'responses': responses,
                    'owner': fix,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = [torch.LongTensor(i) for i in bundle['ids']]
        tids = [torch.LongTensor(i) for i in bundle['tids']]
        context, responses = bundle['context'], bundle['responses']
        return ids, tids, bundle['label'], context, responses, bundle['owner']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        assert len(batch) == 1
        ids, tids, label, context, responses, owner = batch[0]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        label = torch.LongTensor(label)
        mask = generate_mask(ids)
        ids, tids, mask, label = to_cuda(ids, tids, mask, label)
        return {
            'ids': ids, 
            'tids': tids, 
            'mask': mask,
            'label': label,
            'owner': owner,
        }
