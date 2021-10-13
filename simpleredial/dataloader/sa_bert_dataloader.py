from header import *
from .utils import *
from .util_func import *


'''The dataloader are questionable, need to be fixed'''


class SABERTWithNegDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        if 'pj-' in args['model']:
            self.pp_path = f'{os.path.splitext(path)[0]}_pjsa_neg.pt'
        else:
            suffix = args['tokenizer'].replace('/', '_')
            self.pp_path = f'{os.path.splitext(path)[0]}_sa_neg_{suffix}.pt'

        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            if self.args['mode'] == 'train':
                self.extract_by_gray_num()
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data, responses = read_json_data(path, lang=self.args['lang'])
        self.data = []
        if self.args['mode'] == 'train':
            for context, response, candidates in tqdm(data):
                if len(candidates) < 10:
                    candidates += random.sample(responses, 10-len(candidates))
                else:
                    candidates = candidates[:10]
                ids, tids, sids = [], [], []
                for idx, neg in enumerate([response] + candidates):
                    utterances = context + [neg]
                    ids_, tids_, sids_ = self.annotate(utterances)
                    ids.append(ids_)
                    tids.append(tids_)
                    sids.append(sids_)
                self.data.append({
                    'label': [1] + [0] * 10, 
                    'ids': ids, 
                    'tids': tids, 
                    'sids': sids
                })
                self.extract_by_gray_num()
        else:
            for context, response, candidates in tqdm(data):
                # we only need 10 candidates, pos:neg = 1:9
                # compatible with the douban, ecommerce, ubuntu-v1 corpus
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                ids, tids, sids = [],[], []
                for neg in [response] + candidates:
                    utterances = context + [neg]
                    ids_, tids_, sids_ = self.annotate(utterances)
                    ids.append(ids_)
                    tids.append(tids_)
                    sids.append(sids_)
                self.data.append({
                    'label': [1] + [0] * 9, 
                    'ids': ids, 
                    'tids': tids, 
                    'sids': sids,
                })

    def extract_by_gray_num(self):
        # process self.data (after loaded)
        num = self.args['gray_cand_num']
        dataset = []
        for sample in tqdm(self.data):
            dataset.append({
                'label': 1,
                'ids': sample['ids'][0],
                'tids': sample['tids'][0],
                'sids': sample['sids'][0],
            })
            # neg
            neg_idx = random.sample(range(1, 11), num)
            neg_ids = [sample['ids'][i] for i in neg_idx]
            neg_tids = [sample['tids'][i] for i in neg_idx]
            neg_sids = [sample['sids'][i] for i in neg_idx]
            for i, j, k in zip(neg_ids, neg_tids, neg_sids):
                dataset.append({
                    'label': 0,
                    'ids': i,
                    'tids': j,
                    'sids': k,
                })
        self.data = dataset
        # shuffle
        random.shuffle(self.data)

    def annotate(self, utterances):
        tokens = [self.vocab.tokenize(utt) for utt in utterances]
        ids, tids, sids, tcache, scache, l = ['[CLS]'], [0], [0], 0, 0, len(tokens)
        for idx, tok in enumerate(tokens):
            if idx < l - 1:
                ids.extend(tok)
                ids.append('[SEP]')
                tids.extend([tcache] * (len(tok) + 1))
                sids.extend([scache] * (len(tok) + 1))
                scache = 0 if scache == 1 else 1
                tcache = 0
            else:
                tcache = 1
                ids.extend(tok)
                tids.extend([tcache] * len(tok))
                sids.extend([scache] * len(tok))
        ids.append('[SEP]')
        tids.append(tcache)
        sids.append(scache)
        ids = self.vocab.encode(ids, add_special_tokens=False)
        ids, tids, sids = length_limit(ids, self.args['max_len']), length_limit(tids, self.args['max_len']), length_limit(sids, self.args['max_len'])
        assert len(ids) == len(ids) and len(ids) == len(tids)
        return ids, tids, sids

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            tids = torch.LongTensor(bundle['tids'])
            sids = torch.LongTensor(bundle['sids'])
            return ids, tids, sids, bundle['label']
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            sids = [torch.LongTensor(i) for i in bundle['sids']]
            return ids, tids, sids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')

    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            tids = [i[1] for i in batch]
            sids = [i[2] for i in batch]
            label = [i[3] for i in batch]
        else:
            ids, tids, sids, label = [], [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                sids.extend(b[2])
                label.extend(b[3])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        label = torch.LongTensor(label)
        ids, tids, sids, mask, label = to_cuda(ids, tids, sids, mask, label)
        return {
            'ids': ids, 
            'tids': tids, 
            'sids': sids, 
            'mask': mask, 
            'label': label
        }


# ========== SABERT FT Dataset ========== #
class SABERTFTDataset(Dataset):

    '''TODO: truncate the response and context by the max_len; for restoration-200k corpus, it is safe.'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.cls, self.sep = self.vocab.convert_tokens_to_ids(['[CLS]', '[SEP]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_sa_ft_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                ids, tids, sids = self.annotate(utterances)
                self.data.append({'label': label, 'ids': ids, 'tids': tids, 'sids': sids})
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                ids_, tids_, sids_ = [], [], []
                for j in batch:
                    ids, tids, sids = self.annotate(j[1])
                    ids_.append(ids)
                    tids_.append(tids)
                    sids_.append(sids)
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids_,
                    'tids': tids_,
                    'sids': sids_,
                })

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

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            tids = torch.LongTensor(bundle['tids'])
            sids = torch.LongTensor(bundle['sids'])
            label = bundle['label']
            return ids, tids, sids, label
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            tids = [torch.LongTensor(i) for i in bundle['tids']]
            sids = [torch.LongTensor(i) for i in bundle['sids']]
            return ids, tids, sids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')

    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, tids, sids, label = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch]
        else:
            # batch size is batch_size * 10
            ids, tids, sids, label = [], [], [], []
            for b in batch:
                ids.extend(b[0])
                tids.extend(b[1])
                sids.extend(b[2])
                label.extend(b[3])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        label = torch.LongTensor(label)
        ids, tids, sids, mask, label = to_cuda(ids, tids, sids, mask, label)
        return {
            'ids': ids, 
            'tids': tids, 
            'sids': sids, 
            'mask': mask, 
            'label': label
        }
