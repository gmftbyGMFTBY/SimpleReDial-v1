from header import *
from .utils import *
from .util_func import *


class BERTFTCompDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.topk = args['gray_cand_num']
        self.num_labels = args['num_labels']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_comp_plus_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.responses = torch.load(self.pp_path)
            else:
                self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        self.data = []
        if self.args['mode'] == 'train':
            # path = f'{os.path.split(path)[0]}/train_dpr_gray.txt'
            # data = read_dpr_gray(path)
            path = f'{os.path.split(path)[0]}/train_bm25_gray.txt'
            data = read_bm25_hard_negative(path)
            responses, response_overlap = [], set()
            for item in tqdm(data):
                context, response, candidates = item['q'], item['r'], item['nr']
                ids = self.vocab.batch_encode_plus(context + [response], add_special_tokens=False)['input_ids']
                cids = []
                sids, cache = [], 0
                for u in ids[:-1]:
                    cids.extend(u + [self.eos])
                    sids.extend([cache] * (len(u) + 1))
                    cache = 1 if cache == 0 else 0
                sids.pop()
                cids.pop()

                if len(cids) == 0:
                    # the empty sequence raise exception
                    continue

                rids = ids[-1]
                responses.append(rids)
                if response not in response_overlap:
                    responses.append(rids)
                    response_overlap.add(response)
                self.data.append({
                    'context': cids,
                    'sids': sids,
                    'response': rids,
                    'candidates': candidates,
                })
            self.responses = responses
        else:
            # if args['dataset'] in ['ubuntu'] and args['mode'] == 'valid':
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # too many validation samples, just sample 1000
            # data = data[:10000]
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                responses = [b[1][-1] for b in batch]
                context = batch[0][1][:-1]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'context': context,
                    'responses': responses,
                })    

    def __len__(self):
        return len(self.data)

    def _packup(self, cids, rids1, rids2, sids=None):
        cids_, rids1_, rids2_ = deepcopy(cids), deepcopy(rids1), deepcopy(rids2)
        sids_ = deepcopy(sids)
        truncate_pair_two_candidates(
            cids_, rids1_, rids2_,
            self.args['max_len'],
            sids=sids_,
        )
        other_speaker = 0 if sids[-1] == 1 else 1
        ids = [self.cls] + cids_ + [self.sep] + rids1_ + [self.sep] + rids2_ + [self.sep]
        sids_ = [sids_[0]] + sids_ + [sids_[-1]] + [other_speaker] * (len(rids1_) + len(rids2_) + 2)
        tids = [0] * (len(cids_) + 2) + [1] * (len(rids1_) + 1) + [0] * (len(rids2_) + 1)
        # if label == 0:
        #     token_labels = [-100] * (len(cids_) + 2) + [0] * (len(rids1_) + 1) + [1] * (len(rids2_) + 1)
        # else:
        #     token_labels = [-100] * (len(cids_) + 2) + [1] * (len(rids1_) + 1) + [0] * (len(rids2_) + 1)
        # assert len(sids_) == len(ids) == len(token_labels)
        assert len(sids_) == len(ids)
        return ids, tids, sids_

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids, rids = bundle['context'], bundle['response']
            speaker_ids = bundle['sids']

            if self.args['no_hard_negative']:
                hrids = random.sample(self.responses, self.topk)
            else:
                if self.topk > len(bundle['candidates']):
                    candidates = bundle['candidates']
                    if candidates:
                        hrids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
                    else:
                        hrids = []
                    hrids += random.sample(self.responses, self.topk - len(candidates))
                else:
                    candidates = random.sample(bundle['candidates'], self.topk)
                    hrids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
            # context session hard negative samples
            # candidates = bundle['context_session']
            # hrids2 = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
            # if self.topk > len(candidates):
            #     hrids2 += random.sample(self.responses, self.topk - len(hrids2))

            ids, sids, tids, label, token_label = [], [], [], [], []

            # label 0/1: positive vs. easy negative
            for _ in range(self.topk):
                e = random.choice(self.responses)
                if random.random() > 0.5:
                    ids_, tids_, sids_ = self._packup(cids, rids, e, sids=speaker_ids)
                    l = 1
                else:
                    ids_, tids_, sids_ = self._packup(cids, e, rids, sids=speaker_ids)
                    l = 0
                ids.append(ids_)
                sids.append(sids_)
                tids.append(tids_)
                label.append(l)
            # label 0/1: positive negatives vs. bm25 hard negative
            for _ in range(self.topk):
                h = random.choice(hrids)
                if random.random() > 0.5:
                    ids_, tids_, sids_ = self._packup(cids, rids, h, sids=speaker_ids)
                    l = 1
                else:
                    ids_, tids_, sids_ = self._packup(cids, h, rids, sids=speaker_ids)
                    l = 0
                ids.append(ids_)
                sids.append(sids_)
                tids.append(tids_)
                label.append(l)
            # label 0/1: hard negative from the session
            # for h in hrids2:
            #     if random.random() > 0.5:
            #         ids_, tids_, sids_ = self._packup(cids, rids, h, sids=speaker_ids)
            #         l = 1
            #     else:
            #         ids_, tids_, sids_ = self._packup(cids, h, rids, sids=speaker_ids)
            #         l = 0
            #     ids.append(ids_)
            #     sids.append(sids_)
            #     tids.append(tids_)
            #     label.append(l)
            # whole samples
            ids = [torch.LongTensor(i) for i in ids]
            sids = [torch.LongTensor(i) for i in sids]
            tids = [torch.LongTensor(i) for i in tids]
            return ids, sids, tids, label
        else:
            # test
            return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        if self.args['mode'] == 'train':
            data = torch.save((self.data, self.responses), self.pp_path)
        else:
            data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, sids, tids, label = [], [], [], []
            for b in batch:
                ids.extend(b[0])
                sids.extend(b[1])
                tids.extend(b[2])
                label.extend(b[3])
            label = torch.LongTensor(label)
            return {
                'ids': ids, 
                'sids': sids,
                'tids': tids, 
                'label': label
            }
        else:
            # test or valid set
            assert len(batch) == 1
            return {
                'context': batch[0][0],
                'responses': batch[0][1],
                'label': batch[0][2],
            }


class BERTFTCompEvaluationDataset(Dataset):
    
    '''Compare the evaluation results of the generated responses from two systems'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')

        # 22335: bm25+BERT-FP; 22336: dual-bert
        ports = args['file_tags'].split(',')
        path1 = f'{os.path.split(path)[0]}/test_api_pipeline_{ports[0]}_log.txt'
        path2 = f'{os.path.split(path)[0]}/test_api_pipeline_{ports[1]}_log.txt'
        print(f'[!] load file from:\n {path1}\n {path2}')
        
        data1 = read_text_data_from_log_file(path1, lang=args['lang'])
        data2 = read_text_data_from_log_file(path2, lang=args['lang'])

        self.data = []
        for (ctx1, res1), (ctx2, res2) in zip(data1, data2):
            assert ctx1 == ctx2
            self.data.append({
                'context': [i.strip() for i in ctx1.split(' [SEP] ')],
                'responses': [res1, res2]
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        return self.data[i]

    def collate(self, batch):
        assert len(batch) == 1
        return batch[0]


class BERTFTCompMultiDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.topk = args['gray_cand_num']
        self.compare_set_size = args['compare_set_size']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_comp_multi_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.responses = torch.load(self.pp_path)
            else:
                self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        self.data = []
        if self.args['mode'] == 'train':
            path = f'{os.path.split(path)[0]}/train_bm25_gray.txt'
            data = read_bm25_hard_negative(path)
            responses, response_overlap = [], set()
            for item in tqdm(data):
                context, response, candidates = item['q'], item['r'], item['nr']
                ids = self.vocab.batch_encode_plus(context + [response], add_special_tokens=False)['input_ids']
                cids = []
                sids, cache = [], 0
                for u in ids[:-1]:
                    cids.extend(u + [self.eos])
                    sids.extend([cache] * (len(u) + 1))
                    cache = 1 if cache == 0 else 0
                sids.pop()
                cids.pop()

                if self.args['no_inner_session_negative'] is False:
                    candidates += context

                if len(cids) == 0:
                    continue

                rids = ids[-1]
                responses.append(rids)
                if response not in response_overlap:
                    responses.append(rids)
                    response_overlap.add(response)
                self.data.append({
                    'context': cids,
                    'sids': sids,
                    'response': rids,
                    'candidates': candidates,
                })
            self.responses = responses
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                responses = [b[1][-1] for b in batch]
                context = batch[0][1][:-1]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'context': context,
                    'responses': responses,
                })    

    def __len__(self):
        return len(self.data)

    def _packup(self, cids, sids, rids, label):
        ctx_max_length, res_max_length = self.args['ctx_max_length'], self.args['res_max_length']
        num = len(rids)
        # length limitation
        rids = [i[:(res_max_length-2)] for i in rids]
        cids = cids[-(ctx_max_length-2):]
        sids = sids[-(ctx_max_length-2):]

        cids_ = [self.cls] + cids + [self.sep]
        sids_ = [sids[0]] + sids + [sids[-1]]
        tids_ = [0] * (len(cids) + 2)
        lids_ = [-100] * (len(cids) + 2)
        other_speaker = 0 if sids[-1] == 1 else 1
        tcache = 1
        # concatenation
        for idx, (r, l) in enumerate(zip(rids, label)):
            # [unused1] ~ [unused10]
            cids_ += [idx + 1] + r + [self.sep]
            sids_ += [other_speaker] * (len(r) + 2)
            tids_ += [tcache] * (len(r) + 2)
            lids_ += [l] + [-100] * (len(r) + 1)
            # tcache = 0 if tcache == 1 else 1
        assert len(cids_) == len(sids_) == len(tids_) == len(lids_)
        return cids_, sids_, tids_, lids_

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids, rids, sids = deepcopy(bundle['context']), deepcopy(bundle['response']), deepcopy(bundle['sids'])

            if self.args['no_hard_negative']:
                hrids = random.sample(self.responses, self.topk)
            else:
                candidates = random.sample(
                    bundle['candidates'], self.topk
                )
                hrids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
            
            rids = [rids] + random.sample(hrids, self.topk) + random.sample(self.responses, self.compare_set_size - self.topk - 1)
            random_idx = list(range(self.compare_set_size))
            random.shuffle(random_idx)
            label = [1] + [0] * (self.compare_set_size - 1)
            label = [label[i] for i in random_idx]
            cls_label = random_idx.index(0)
            rids  = [rids[i] for i in random_idx]
            ids, sids, tids, lids = self._packup(cids, sids, rids, label) 
            ids = torch.LongTensor(ids)
            sids = torch.LongTensor(sids)
            tids = torch.LongTensor(tids)
            lids = torch.LongTensor(lids)
            return ids, sids, tids, lids, cls_label
        else:
            # test
            return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        if self.args['mode'] == 'train':
            data = torch.save((self.data, self.responses), self.pp_path)
        else:
            data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, sids, tids, lids, label = [], [], [], [], []
            for a, b, c, d, e in batch:
                ids.append(a)
                sids.append(b)
                tids.append(c)
                lids.append(d)
                label.append(e)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            lids = pad_sequence(lids, batch_first=True, padding_value=-100)
            label = torch.LongTensor(label)
            mask = generate_mask(ids)
            ids, sids, tids, lids, mask, label = to_cuda(ids, sids, tids, lids, mask, label)
            return {
                'ids': ids, 
                'sids': sids,
                'tids': tids, 
                'lids': lids,
                'mask': mask,
                'label': label,
            }
        else:
            # test or valid set
            assert len(batch) == 1
            return {
                'context': batch[0][0],
                'responses': batch[0][1],
                'label': batch[0][2],
            }

            
class BERTFTCompMultiCLSDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.topk = args['gray_cand_num']
        self.compare_set_size = args['compare_set_size']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_comp_multi_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.responses = torch.load(self.pp_path)
            else:
                self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        self.data = []
        if self.args['mode'] == 'train':
            path = f'{os.path.split(path)[0]}/train_bm25_gray.txt'
            data = read_bm25_hard_negative(path)
            responses, response_overlap = [], set()
            for item in tqdm(data):
                context, response, candidates = item['q'], item['r'], item['nr']
                ids = self.vocab.batch_encode_plus(context + [response], add_special_tokens=False)['input_ids']
                cids = []
                sids, cache = [], 0
                for u in ids[:-1]:
                    cids.extend(u + [self.eos])
                    sids.extend([cache] * (len(u) + 1))
                    cache = 1 if cache == 0 else 0
                sids.pop()
                cids.pop()

                if self.args['no_inner_session_negative'] is False:
                    candidates += context

                if len(cids) == 0:
                    continue

                rids = ids[-1]
                responses.append(rids)
                if response not in response_overlap:
                    responses.append(rids)
                    response_overlap.add(response)
                self.data.append({
                    'context': cids,
                    'sids': sids,
                    'response': rids,
                    'candidates': candidates,
                })
            self.responses = responses
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                responses = [b[1][-1] for b in batch]
                context = batch[0][1][:-1]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'context': context,
                    'responses': responses,
                })    

    def __len__(self):
        return len(self.data)

    def _packup(self, cids, sids, rids):
        ctx_max_length, res_max_length = self.args['ctx_max_length'], self.args['res_max_length']
        num = len(rids)
        # length limitation
        rids = [i[:(res_max_length-2)] for i in rids]
        cids = cids[-(ctx_max_length-2):]
        sids = sids[-(ctx_max_length-2):]

        cids_ = [self.cls] + cids + [self.sep]
        sids_ = [sids[0]] + sids + [sids[-1]]
        tids_ = [0] * (len(cids) + 2)
        other_speaker = 0 if sids[-1] == 1 else 1
        tcache = 1
        # concatenation
        for idx, r in enumerate(rids):
            # [unused1] ~ [unused10]
            cids_ += [idx + 1] + r + [self.sep]
            sids_ += [other_speaker] * (len(r) + 2)
            tids_ += [tcache] * (len(r) + 2)
            tcache = 0 if tcache == 1 else 1
        assert len(cids_) == len(sids_) == len(tids_)
        return cids_, sids_, tids_

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids, rids, sids = deepcopy(bundle['context']), deepcopy(bundle['response']), deepcopy(bundle['sids'])

            if self.args['no_hard_negative']:
                hrids = random.sample(self.responses, self.topk)
            else:
                candidates = random.sample(
                    bundle['candidates'], self.topk
                )
                hrids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
            
            rids = [rids] + random.sample(hrids, self.topk) + random.sample(self.responses, self.compare_set_size - self.topk - 1)
            random_idx = list(range(self.compare_set_size))
            random.shuffle(random_idx)
            label = random_idx.index(0)
            rids  = [rids[i] for i in random_idx]
            ids, sids, tids = self._packup(cids, sids, rids) 
            ids = torch.LongTensor(ids)
            sids = torch.LongTensor(sids)
            tids = torch.LongTensor(tids)
            return ids, sids, tids, label
        else:
            # test
            return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        if self.args['mode'] == 'train':
            data = torch.save((self.data, self.responses), self.pp_path)
        else:
            data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, sids, tids, label = [], [], [], []
            for a, b, c, d in batch:
                ids.append(a)
                sids.append(b)
                tids.append(c)
                label.append(d)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
            tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
            label = torch.LongTensor(label)
            mask = generate_mask(ids)
            ids, sids, tids, label, mask = to_cuda(ids, sids, tids, label, mask)
            return {
                'ids': ids, 
                'sids': sids,
                'tids': tids, 
                'label': label,
                'mask': mask,
            }
        else:
            # test or valid set
            assert len(batch) == 1
            return {
                'context': batch[0][0],
                'responses': batch[0][1],
                'label': batch[0][2],
            }

            
class BERTFTCompTokenDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.topk = args['gray_cand_num']
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_comp_token_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.responses = torch.load(self.pp_path)
            else:
                self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        self.data = []
        if self.args['mode'] == 'train':
            path = f'{os.path.split(path)[0]}/train_bm25_gray.txt'
            data = read_bm25_hard_negative(path)
            responses, response_overlap = [], set()
            for item in tqdm(data):
                context, response, candidates = item['q'], item['r'], item['nr']
                ids = self.vocab.batch_encode_plus(context + [response], add_special_tokens=False)['input_ids']
                cids = []
                sids, cache = [], 0
                for u in ids[:-1]:
                    cids.extend(u + [self.eos])
                    sids.extend([cache] * (len(u) + 1))
                    cache = 1 if cache == 0 else 0
                sids.pop()
                cids.pop()

                if len(cids) == 0:
                    continue

                rids = ids[-1]
                responses.append(rids)
                if response not in response_overlap:
                    responses.append(rids)
                    response_overlap.add(response)
                self.data.append({
                    'context': cids,
                    'sids': sids,
                    'response': rids,
                    'candidates': candidates,
                })
            self.responses = responses
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                responses = [b[1][-1] for b in batch]
                context = batch[0][1][:-1]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'context': context,
                    'responses': responses,
                })    

    def __len__(self):
        return len(self.data)

    def _packup(self, cids, sids, rids1, rids2, label1, label2):
        cids_, sids_, rids1_, rids2_ = deepcopy(cids), deepcopy(sids), deepcopy(rids1), deepcopy(rids2)
        truncate_pair_two_candidates(
            cids_, rids1_, rids2_,
            self.args['max_len'],
            sids=sids_,
        )
        other_speaker = 0 if sids_[-1] == 1 else 1
        cids__ = [self.cls] + cids_ + [self.sep] + [1] + rids1_ + [self.sep] + [2] + rids2_ + [self.sep]
        sids__ = [sids_[0]] + sids_ + [sids_[-1]] + [other_speaker] * (len(rids1_) + len(rids2_) + 4)
        tids__ = [0] * (len(cids_) + 2) + [1] * (len(rids1_) + 2) + [0] * (len(rids2_) + 2)
        tlids__ = [-100] * (len(cids_) + 2) + [label1] + [-100] * (len(rids1_) + 1) + [label2] + [-100] * (len(rids2_) + 1)
        assert len(tids__) == len(sids__) == len(cids__) == len(tlids__)
        return cids__, tids__, sids__, tlids__

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids, rids = bundle['context'], bundle['response']
            speaker_ids = bundle['sids']

            if self.args['no_hard_negative']:
                hrids = random.sample(self.responses, self.topk)
            else:
                if self.topk > len(bundle['candidates']):
                    candidates = bundle['candidates']
                    if candidates:
                        hrids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
                    else:
                        hrids = []
                    hrids += random.sample(self.responses, self.topk - len(candidates))
                else:
                    candidates = random.sample(bundle['candidates'], self.topk)
                    hrids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']

            ids, sids, tids, tlids = [], [], [], []

            # positive vs. easy negative
            for _ in range(self.topk):
                e = random.choice(self.responses)
                if random.random() > 0.5:
                    ids_, tids_, sids_, tlids_ = self._packup(cids, speaker_ids, rids, e, 1, 0)
                else:
                    ids_, tids_, sids_, tlids_ = self._packup(cids, speaker_ids, e, rids, 0, 1)
                ids.append(ids_)
                sids.append(sids_)
                tids.append(tids_)
                tlids.append(tlids_)
            # positive negatives vs. bm25 hard negative
            for _ in range(self.topk):
                h = random.choice(hrids)
                if random.random() > 0.5:
                    ids_, tids_, sids_, tlids_ = self._packup(cids, speaker_ids, rids, h, 1, 0)
                else:
                    ids_, tids_, sids_, tlids_ = self._packup(cids, speaker_ids, h, rids, 0, 1)
                ids.append(ids_)
                sids.append(sids_)
                tids.append(tids_)
                tlids.append(tlids_)
            # easy neg vs. easy neg.
            for _ in range(self.topk):
                e1, e2 = random.sample(self.responses, 2)
                ids_, tids_, sids_, tlids_ = self._packup(cids, speaker_ids, e1, e2, 0, 0)
                ids.append(ids_)
                sids.append(sids_)
                tids.append(tids_)
                tlids.append(tlids_)
            # whole samples
            ids = [torch.LongTensor(i) for i in ids]
            sids = [torch.LongTensor(i) for i in sids]
            tids = [torch.LongTensor(i) for i in tids]
            tlids = [torch.LongTensor(i) for i in tlids]
            return ids, sids, tids, tlids
        else:
            # test
            return bundle['context'], bundle['responses'], bundle['label']

    def save(self):
        if self.args['mode'] == 'train':
            data = torch.save((self.data, self.responses), self.pp_path)
        else:
            data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, sids, tids, tlids = [], [], [], []
            for b in batch:
                ids.extend(b[0])
                sids.extend(b[1])
                tids.extend(b[2])
                tlids.extend(b[3])
            return {
                'ids': ids, 
                'sids': sids,
                'tids': tids, 
                'tlids': tlids,
            }
        else:
            # test or valid set
            assert len(batch) == 1
            return {
                'context': batch[0][0],
                'responses': batch[0][1],
                'label': batch[0][2],
            }
