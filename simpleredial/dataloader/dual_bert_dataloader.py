from header import *
from .utils import *
from .util_func import *
from .augmentation import *


class BERTDualDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances(path, lang=self.args['lang'])

        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }


class BERTDualO2MDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.gray_num = args['gray_cand_num']
       
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_gray_o2m_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        
        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_one2many(path, lang=self.args['lang'])
            for context, response in tqdm(data):
                ctx1 = self.vocab.batch_encode_plus(context[0], add_special_tokens=False)['input_ids']
                ids1 = []
                for u in ctx1:
                    ids1.extend(u + [self.sep])
                ids1.pop()
                ids1 = ids1[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids1 = [self.cls] + ids1 + [self.sep]

                ctx2 = self.vocab.batch_encode_plus(context[1], add_special_tokens=False)['input_ids']
                ids2 = []
                for u in ctx2:
                    ids2.extend(u + [self.sep])
                ids2.pop()
                ids2 = ids2[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids2 = [self.cls] + ids2 + [self.sep]

                rids = self.vocab.encode(response, add_special_tokens=False)
                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'context': [ids1, ids2],
                    'response': rids,
                })
        else:
            data = read_text_data_utterances(path, lang='zh')
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    rids.append([self.cls] + item[-1] + [self.sep])
                ids = [self.cls]
                for u in item[:-1]:
                    ids.extend(u + [self.sep])
                ids[-1] = self.sep
                ids = length_limit(ids, self.args['max_len'])
                rids = [length_limit_res(i, self.args['res_max_len'], sep=self.sep) for i in rids]
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids, 
                    'rids': rids,
                })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            rids = torch.LongTensor(bundle['response'])
            ids = [torch.LongTensor(i) for i in bundle['context']]
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            rids = [i[1] for i in batch]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            rids, rids_mask = to_cuda(rids, rids_mask)
            ids, ids_mask = [], []
            for i in range(2):
                ids_ = [item[0][i] for item in batch]
                ids_ = pad_sequence(ids_, batch_first=True, padding_value=self.pad)
                ids_mask_ = generate_mask(ids_)
                ids_, ids_mask_ = to_cuda(ids_, ids_mask_)
                ids.append(ids_)
                ids_mask.append(ids_mask_)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
            }
        else:
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label
            }

            
class BERTDualFullDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])

            # path = f'{os.path.split(path)[0]}/train_full_filter_{args["data_filter_size"]}.pt'
            # data = read_text_data_utterances_full_data_filter(path)

            # data = read_text_data_utterances_full_large(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'], min_length=self.args['min_conv_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            # data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullPseudoDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_pseudo_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            pseudo_path = f'{os.path.splitext(path)[0]}_gray.txt'
            data = read_text_data_utterances_and_full_and_pesudo_pairs(path, pseudo_path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullExtraNegDataset(Dataset):

    '''add some extra negative samples for training'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.extra_neg = args['ext_neg_size']

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_extra_neg_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None


        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': utterances[:-1],
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext'], i
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            index = [i[4] for i in batch]
            # extra negative samples
            while True:
                random_idx = random.sample(range(len(self.data)), self.extra_neg)
                if len(set(index) & set(random_idx)) == 0:
                    break
            erids = [torch.LongTensor(self.data[i]['rids']) for i in random_idx]    # [M]
            rids += erids    # [B+M]

            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualSemiDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_semi_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.ext_data = torch.load(self.pp_path)
            else:
                self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
            # extended unsupervised utterances
            if args['dataset'] in ['restoration-200k', 'douban']:
                ext_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
            else:
                ext_path = f'{args["root_dir"]}/data/{args["dataset"]}/ext_corpus.txt'
            ext_data = read_extended_douban_corpus(ext_path)
            self.ext_data = []
            inner_bsz = 256
            for idx in tqdm(range(0, len(ext_data), inner_bsz)):
                utterances = ext_data[idx:idx+inner_bsz]
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                rids = [[self.cls] + i[:self.args['res_max_len']-2] + [self.sep] for i in item]
                self.ext_data.extend(rids)
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        if self.args['mode'] == 'train':
            torch.save((self.data, self.ext_data), self.pp_path)
        else:
            torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]

            # extended dataset
            ext_rids = random.sample(self.ext_data, len(batch)*self.args['ext_num'])
            ext_rids = [torch.LongTensor(i) for i in ext_rids]
            ext_rids = pad_sequence(ext_rids, batch_first=True, padding_value=self.pad)
            ext_rids_mask = generate_mask(ext_rids)

            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            rids_mask = generate_mask(rids)
            ids, rids, ext_rids, ids_mask, rids_mask, ext_rids_mask = to_cuda(ids, rids, ext_rids, ids_mask, rids_mask, ext_rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ext_rids': ext_rids,
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ext_rids_mask': ext_rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullWithSelfPlayDataset(Dataset):

    '''train_gray.txt must be generated by the full conversation context'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_with_self_play_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'])
            # self-play corpus
            ext_data = read_json_data_dual_bert_full(f'{args["root_dir"]}/data/{args["dataset"]}/train_gen.txt')
            data += ext_data
            print(f'[!] collect {len(data)} samples for full-with-self-play dataloader')
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullNegSessionDataset(Dataset):

    '''more positive pairs to train the dual bert model, with hard negative samples which are in the same session'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.gray_cand_num = args['gray_cand_num']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_neg_session_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full_neg_session(path, lang=self.args['lang'])
            for label, utterances, neg in tqdm(data):
                if label == 0:
                    continue
                self.data.append({
                    'utterances': utterances,
                    'neg': neg,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            utterances, neg = bundle['utterances'], bundle['neg']
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            cids, rids = item[:-1], item[-1]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
            rids = rids[:(self.args['res_max_len']-2)]
            ids = [self.cls] + ids + [self.sep]
            rids = [self.cls] + rids + [self.sep]
            # neg inner nession
            hrids = self.vocab.batch_encode_plus(random.sample(neg, self.gray_cand_num), add_special_tokens=False)['input_ids']
            hrids = [[self.cls] + i[:(self.args['res_max_len']-2)] + [self.sep] for i in hrids]
            ids = torch.LongTensor(ids)
            rids = torch.LongTensor(rids)
            hrids = [torch.LongTensor(i) for i in hrids]
            return ids, rids, hrids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = [i[1] for i in batch]
            hrids = []
            for i in batch:
                hrids.extend(i[2])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            hrids = pad_sequence(hrids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            hrids_mask = generate_mask(hrids)
            ids, rids, hrids, ids_mask, rids_mask, hrids_mask = to_cuda(ids, rids, hrids, ids_mask, rids_mask, hrids_mask)
            return {
                'ids': ids, 
                'rids': rids,
                'hrids': hrids,    # [B*K, S]
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'hrids_mask': hrids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualHNDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.gray_cand_num = args['gray_cand_num']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_hn_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = torch.load(f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_simcse.pt')
            for key in tqdm(data):
                value = data[key]
                utterances = [i['text'] for i in value]
                if len(utterances) <= 2:
                    candidates = list(chain(*[i['cands'] for i in value]))
                else:
                    candidates = list(chain(*[i['cands'] for i in value[:-2]]))
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'cands': candidates,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids, rids = bundle['ids'], bundle['rids']
            random_idx = random.sample(len(bundle['cands']), self.gray_cand_num)
            candidates = [bundle['cands'][i] for i in random_idx]
            # delete to make sure more hard negative samples can be used
            for i in candidates:
                self.data[i]['cands'].remove(i)
            # neg inner nession
            hrids = self.vocab.batch_encode_plus(candidates, add_special_tokens=False)['input_ids']
            hrids = [[self.cls] + i[:(self.args['res_max_len']-2)] + [self.sep] for i in hrids]
            ids = torch.LongTensor(ids)
            rids = torch.LongTensor(rids)
            hrids = [torch.LongTensor(i) for i in hrids]
            return ids, rids, hrids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = [i[1] for i in batch]
            hrids = []
            for i in batch:
                hrids.extend(i[2])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            hrids = pad_sequence(hrids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            hrids_mask = generate_mask(hrids)
            ids, rids, hrids, ids_mask, rids_mask, hrids_mask = to_cuda(ids, rids, hrids, ids_mask, rids_mask, hrids_mask)
            return {
                'ids': ids, 
                'rids': rids,
                'hrids': hrids,
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'hrids_mask': hrids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualExtFullDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_ext_dialog_full_{suffix}.pt'
        print(f'[!] full turn length: {self.args["full_turn_length"]}')
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            path = f'{os.path.splitext(path)[0]}_gen_ext.txt'
            data = read_text_data_ext_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

class BERTDualFullDACTXDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_da_ctx_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full_da_ctx(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'], da_ctx_num=args['da_ctx_num'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

class BERTDualReplaceBadResponseDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.replace_ratio = args['replace_ratio']

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_replace_bad_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_one2many_replace(path, lang=self.args['lang'])
            for q, r, cands, bad_response in tqdm(data):
                item = self.vocab.batch_encode_plus(q, add_special_tokens=False)['input_ids']
                ids = []
                for u in item:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]
                item = self.vocab.batch_encode_plus([r]+cands, add_special_tokens=False)['input_ids']
                rids = [[self.cls] + i[:(self.args['res_max_len']-2)] + [self.sep] for i in item]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'bad_response': bad_response
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            if bundle['bad_response']:
                # if random.random() < self.replace_ratio:
                #     rids = torch.LongTensor(random.choice(bundle['rids']))
                # else:
                #     rids = torch.LongTensor(bundle['rids'][0])
                rids = torch.LongTensor(random.choice(bundle['rids']))
            else:
                rids = torch.LongTensor(bundle['rids'][0])
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullExtraNegFromOutDatasetDataset(Dataset):

    '''add some extra negative samples for training'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.extra_neg = args['extra_neg']

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_extra_neg_from_outdataset_{suffix}.pt'
        if os.path.exists(self.pp_path):
            if self.args['mode'] == 'train':
                self.data, self.ext_data = torch.load(self.pp_path)
            else:
                self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None


        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': utterances[:-1],
                    'rtext': utterances[-1],
                })
            ext_path = f'{args["root_dir"]}/data/ext_douban/train.txt'
            ext_data = read_extended_douban_corpus(ext_path)
            self.ext_data = []
            for utterance in tqdm(ext_data):
                ids = self.vocab.encode(utterance, add_special_tokens=False)
                ids = ids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                self.ext_data.append({
                    'ids': ids,
                    'text': utterance
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        if self.args['mode'] == 'train':
            torch.save((self.data, self.ext_data), self.pp_path)
        else:
            torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            # extra negative samples
            random_idx = random.sample(range(len(self.ext_data)), self.extra_neg)
            erids = [torch.LongTensor(self.ext_data[i]['ids']) for i in random_idx]    # [M]
            rids += erids    # [B+M]

            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualBertMaskHardNegativeDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_bert_mask_hn_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            path = f'{os.path.splitext(path)[0]}_bert_mask_da_results.pt'
            print(f'[!] prepare to load data from {path}')
            data = read_torch_data_bert_mask(path, hard_negative_num=self.args['gray_cand_num'])
            pool = list(set([i[1] for i in data]))
            for c, r, cand in tqdm(data):
                item = self.vocab.batch_encode_plus(c+[r], add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                if len(cand) < self.args['gray_cand_num']:
                    cand += random.sample(pool, self.args['gray_cand_num']-len(cand))
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(c),
                    'rtext': r,
                    'cands': cand,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            cands = random.sample(bundle['cands'], self.args['gray_cand_num'])
            cands = self.vocab.batch_encode_plus(cands, add_special_tokens=False)['input_ids']
            hrids = [torch.LongTensor([self.cls] + i[:(self.args['res_max_len']-2)] + [self.sep]) for i in cands]
            rids = [rids] + hrids
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = []
            for i in batch:
                rids.extend(i[1])
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullWithHardNegDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_with_hard_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_json_data_from_gray_file(f'{os.path.splitext(path)[0]}_gray.txt')
            for c, r, hn in tqdm(data):
                item = self.vocab.batch_encode_plus(c+[r], add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(c),
                    'rtext': r,
                    'cands': hn,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            candidates = random.sample(bundle['cands'], self.args['gray_cand_num'])
            candidates = [[self.cls] + self.vocab.encode(i)[:self.args['res_max_len']-2]+[self.sep] for i in candidates]
            candidates = [torch.LongTensor(i) for i in candidates]
            return ids, [rids]+candidates, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = []
            for i in batch:
                rids.extend(i[1])
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullWithPositionWeightDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.special_tokens = set([self.sep, self.cls, self.eos, self.unk])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_pos_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
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
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                position_w = [w-self.args['w_delta']] + position_w + [self.args['w_sp_token']]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'position_w': position_w,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'position_w': position_w,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            position_w = torch.tensor(bundle['position_w'])
            return ids, rids, position_w, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            position_w = torch.tensor(bundle['position_w'])
            return ids, rids, position_w, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            pos_w = [i[2] for i in batch]
            ctext = [i[3] for i in batch]
            rtext = [i[4] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            pos_w = pad_sequence(pos_w, batch_first=True, padding_value=0.)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, pos_w, ids_mask, rids_mask = to_cuda(ids, rids, pos_w, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'pos_w': pos_w,
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, pos_w, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, pos_w, rids_mask, label = to_cuda(ids, rids, pos_w, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'pos_w': pos_w,
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualBertMaskHardNegativeWithPositionWeightDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.special_tokens = set([self.sep, self.cls, self.unk])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_bert_mask_hn_pos_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            path = f'{os.path.splitext(path)[0]}_bert_mask_da_results.pt'
            print(f'[!] prepare to load data from {path}')
            data = read_torch_data_bert_mask(path, hard_negative_num=self.args['gray_cand_num'])
            pool = list(set([i[1] for i in data]))
            for c, r, cand in tqdm(data):
                item = self.vocab.batch_encode_plus(c+[r], add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
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
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                position_w = [w-self.args['w_delta']] + position_w + [self.args['w_sp_token']]
                rids = [self.cls] + rids + [self.sep]
                if len(cand) < self.args['gray_cand_num']:
                    cand += random.sample(pool, self.args['gray_cand_num']-len(cand))
                self.data.append({
                    'ids': ids,
                    'position_w': position_w,
                    'rids': rids,
                    'cands': cand,
                    'ctext': ' [SEP] '.join(c),
                    'rtext': r,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'position_w': position_w,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            position_w = torch.tensor(bundle['position_w'])
            rids = torch.LongTensor(bundle['rids'])
            cands = random.sample(bundle['cands'], self.args['gray_cand_num'])
            cands = self.vocab.batch_encode_plus(cands, add_special_tokens=False)['input_ids']
            hrids = [torch.LongTensor([self.cls] + i[:(self.args['res_max_len']-2)] + [self.sep]) for i in cands]
            rids = [rids] + hrids
            return ids, rids, position_w, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            position_w = torch.tensor(bundle['position_w'])
            return ids, rids, position_w, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            pos_w = [i[2] for i in batch]
            rids = []
            for i in batch:
                rids.extend(i[1])
            ctext = [i[3] for i in batch]
            rtext = [i[4] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            pos_w = pad_sequence(pos_w, batch_first=True, padding_value=0.)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, pos_w, ids_mask, rids_mask = to_cuda(ids, rids, pos_w, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'pos_w': pos_w, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, pos_w, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, pos_w, rids_mask, label = to_cuda(ids, rids, pos_w, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'pos_w': pos_w,
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualSABertMaskHardNegativeDataset(Dataset):

    '''speaker aware bert mask hard negative augmentation for dual-bert model'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_bert_sa_mask_hn_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            path = f'{os.path.splitext(path)[0]}_bert_mask_da_results.pt'
            print(f'[!] prepare to load data from {path}')
            data = read_torch_data_bert_mask(path, hard_negative_num=self.args['gray_cand_num'])
            pool = list(set([i[1] for i in data]))
            for c, r, cand in tqdm(data):
                item = self.vocab.batch_encode_plus(c+[r], add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                # speaker embedding index
                sids, scache = [], 0
                for u in cids:
                    ids.extend(u + [self.sep])
                    sids.extend([scache] * (len(u) + 1))
                    scache = 0 if scache == 1 else 1
                sids.pop()
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                sids = sids[-(self.args['max_len']-2):]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                sids = [0] + sids + [sids[-1]]
                rids = [self.cls] + rids + [self.sep]
                if len(cand) < self.args['gray_cand_num']:
                    cand += random.sample(pool, self.args['gray_cand_num']-len(cand))
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'sids': sids,
                    'ctext': ' [SEP] '.join(c),
                    'rtext': r,
                    'cands': cand,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids = []
                    # speaker embedding index
                    sids, scache = [], 0
                    for u in cids:
                        ids.extend(u + [self.sep])
                        sids.extend([scache] * (len(u) + 1))
                    sids.pop()
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    sids = sids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    sids = [0] + sids + [sids[-1]]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'sids': sids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            sids = torch.LongTensor(bundle['sids'])
            rids = torch.LongTensor(bundle['rids'])
            cands = random.sample(bundle['cands'], self.args['gray_cand_num'])
            cands = self.vocab.batch_encode_plus(cands, add_special_tokens=False)['input_ids']
            hrids = [torch.LongTensor([self.cls] + i[:(self.args['res_max_len']-2)] + [self.sep]) for i in cands]
            rids = [rids] + hrids
            return ids, rids, bundle['ctext'], bundle['rtext'], sids
        else:
            ids = torch.LongTensor(bundle['ids'])
            sids = torch.LongTensor(bundle['sids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text'], sids

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = []
            for i in batch:
                rids.extend(i[1])
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            sids = [i[4] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            sids = pad_sequence(sids, batch_first=True, padding_value=0)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, sids, rids, ids_mask, rids_mask = to_cuda(ids, sids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'sids': sids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text, sids = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, sids, rids, rids_mask, label = to_cuda(ids, sids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'sids': sids,
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualSimCSEHardNegativeDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_simcse_hn_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_simcse.pt'
            data = torch.load(path)
            pool = []
            for key, value in data.items():
                for utterance in value:
                    pool.append(utterance['text'])
            pool = list(set(pool))
            for key, sample in tqdm(data.items()):
                utterances = [i['text'] for i in sample]
                candidates = list(set(chain(*[i['cands'] for i in sample[:-2]])))    # ignore the candidates of the responses
                if len(candidates) < self.args['gray_cand_num']:
                    candidates += random.sample(pool, self.args['gray_cand_num']-len(candidates))
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                    'cands': candidates,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # debug
            # data = data[:10000]
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            random_idx = random.sample(range(len(bundle['cands'])), self.args['gray_cand_num'])
            cands = [bundle['cands'][i] for i in random_idx]
            # delete 
            # for text in cands:
            #     self.data[i]['cands'].remove(text)
            cands = self.vocab.batch_encode_plus(cands, add_special_tokens=False)['input_ids']
            hcids = [torch.LongTensor([self.cls] + i[:(self.args['res_max_len']-2)] + [self.sep]) for i in cands]
            rids = [rids] + hcids
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = []
            for i in batch:
                rids.extend(i[1])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualMaskFullDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.mask_aug_t = args['mask_aug_t']

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_mask_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]

                # context with masked augmentation
                max_ctx_turn, counter = 0, 0
                for u in cids[::-1]:
                    counter += len(u) + 1
                    max_ctx_turn += 1
                    # [CLS] token
                    if counter + 1 > self.args['max_len']:
                        break
                cids_ = cids[-max_ctx_turn:]

                if len(cids_) > 1:
                    random_idx = random.randint(0, max_ctx_turn-2)
                    ids, counter = [], 0
                    for u in cids_:
                        if counter == random_idx:
                            pass
                        else:
                            ids.extend(u + [self.sep])
                        counter += 1
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    ids = [self.cls] + ids + [self.sep]
                    self.data.append({
                        'ids': ids,
                        'rids': rids,
                        'ctext': ' [SEP] '.join([i for idx, i in enumerate(utterances[-max_ctx_turn:-1]) if idx != random_idx]),
                        'rtext': utterances[-1],
                    })
                #
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualSimCSEHNCTXDataset(Dataset):

    '''hard negative for context'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_hn_ctx_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            path = f'{args["root_dir"]}/data/{args["dataset"]}/train_gray_simcse.pt'
            data, data_cands = read_torch_data_simcse(path, lang=self.args['lang'])
            for idx in tqdm(range(len(data))):
                utterances, candidates = data[idx], data_cands[idx]
                # prepare the candidates
                length = len(utterances)
                cands = []
                for i in range(length):
                    if i != length - 2:
                        cands.append(utterances[i])
                        cands.extend(candidates[i])

                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]

                # count the max turn size 
                max_ctx_turn, counter = 0, 0
                for u in cids[::-1]:
                    counter += len(u) + 1
                    max_ctx_turn += 1
                    # add [CLS]
                    if counter + 1 > self.args['max_len']:
                        break
                cids = cids[-max_ctx_turn:]

                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]
                
                ids2 = []
                begin = 0 if len(cids) <= 1 else random.randint(1, len(cids)-1)
                for u in cids[begin:]:
                    ids2.extend(u + [self.sep])
                ids2.pop()
                ids2 = ids2[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids2 = [self.cls] + ids2 + [self.sep]

                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'ids2': ids2,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                    'cands': cands 
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            ids2 = torch.LongTensor(bundle['ids2'])
            hn = random.sample(bundle['cands'], self.args['gray_cand_num'])
            items = self.vocab.batch_encode_plus(hn, add_special_tokens=False)['input_ids']
            hids = []
            for item in items:
                item = item[-(self.args['res_max_len']-2):]
                item = [self.cls] + item + [self.sep]
                hids.append(torch.LongTensor(item))

            rids = torch.LongTensor(bundle['rids'])
            return ids, ids2, hids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, ids1, ids2 = [], [], []
            for i in batch:
                ids.append(i[0])
                ids1.append(i[1])
                ids2.extend(i[2])
            rids = [i[3] for i in batch]
            ctext = [i[4] for i in batch]
            rtext = [i[5] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)    # [B*M, S], where M = gray_cand_num + 2
            ids1 = pad_sequence(ids1, batch_first=True, padding_value=self.pad)    # [B*M, S], where M = gray_cand_num + 2
            ids2 = pad_sequence(ids2, batch_first=True, padding_value=self.pad)    # [B*M, S], where M = gray_cand_num + 2
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            ids1_mask = generate_mask(ids1)
            ids2_mask = generate_mask(ids2)
            rids_mask = generate_mask(rids)
            ids, ids1, ids2, rids, ids_mask, ids1_mask, ids2_mask, rids_mask = to_cuda(ids, ids1, ids2, rids, ids_mask, ids1_mask, ids2_mask, rids_mask)
            return {
                'ids': ids, 
                'ids1': ids1, 
                'ids2': ids2, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'ids1_mask': ids1_mask, 
                'ids2_mask': ids2_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualHNCTX2Dataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        dataset = read_text_data_utterances(path, lang=self.args['lang'])
        data, counter = {}, 0
        for label, utterances in dataset:
            if label == 0:
                continue
            start_num = max(1, len(utterances) - self.args['full_turn_length'])
            data[counter] = []
            for i in range(start_num, len(utterances)):
                data[counter].append((1, utterances[:i+1]))
            counter += 1

        self.data = []
        if self.args['mode'] == 'train':
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            batch = batch[0]
            ids, rids, label = batch[0], batch[1], batch[2]
            text = batch[3]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualMultiCTXFullDataset(Dataset):

    '''more positive pairs to train the dual bert model, random delete one utterance in context'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.multi_ctx = args['multi_ctx']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_multi_ctx_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}: {len(self.data)} samples')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]

                # cut cids
                counter, max_len = 0, 0
                for i in cids[::-1]:
                    counter += len(i) + 1
                    max_len += 1
                    if counter + 1 > self.args['max_len']:
                        break
                cids = cids[-max_len:]

                if len(cids) > 1:
                    # augment
                    utterances_ = utterances[:-1][-max_len:]
                    del_num = min(self.multi_ctx, len(cids)-1)
                    for del_start_idx in range(1, del_num+1):
                        ids = []
                        for u in cids[del_start_idx:]:
                            ids.extend(u + [self.sep])
                        ids.pop()
                        ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                        ids = [self.cls] + ids + [self.sep]
                        self.data.append({
                            'ids': ids,
                            'rids': rids,
                            'ctext': ' [SEP] '.join(utterances_[del_start_idx:]),
                            'rtext': utterances[-1],
                        })

                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]

                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
            # NOTE: debug
            # if args['dataset'] == 'ubuntu':
            #     self.data = random.sample(self.data, 1000)
            #     print(f'[!] only 1000 test samples are used during ubuntu training')
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualRandomFullDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_random_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                rids = rids[:(self.args['res_max_len']-2)]
                self.data.append({
                    'ids': cids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids, rids = bundle['ids'], bundle['rids']
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            # random ctx
            sids = []
            if len(cids) == 1:
                scids = random.choice(self.data)['ids']
                for u in scids:
                    sids.extend(u + [self.sep])
            else:
                scids = deepcopy(cids)
                random_idx = list(range(len(scids)))
                while True:
                    random.shuffle(random_idx)
                    if random_idx[-1] != len(cids) - 1:
                        break
                for i in random_idx:
                    u = scids[i]
                    sids.extend(u + [self.sep])
            sids.pop()

            ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
            sids = sids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
            ids = [self.cls] + ids + [self.sep]
            sids = [self.cls] + sids + [self.sep]
            rids = [self.cls] + rids + [self.sep]

            ids, rids = torch.LongTensor(ids), torch.LongTensor(rids)
            sids = torch.LongTensor(sids)
            return ids, rids, sids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            sids = [i[2] for i in batch]
            ctext = [i[3] for i in batch]
            rtext = [i[4] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            sids_mask = generate_mask(sids)
            ids, rids, sids, sids_mask, ids_mask, rids_mask = to_cuda(ids, rids, sids, sids_mask, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'sids': sids, 
                'sids_mask': sids_mask, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualSAFullDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_sa_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
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
                if len(ids) == 0:
                    continue
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                sids = sids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                tlids = tlids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                sids = [0] + sids + [sids[-1]]
                tlids = [0] + tlids + [tlids[-1]]
                ids = [self.cls] + ids + [self.sep]
                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'sids': sids,
                    'tlids': tlids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            # data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
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
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    sids = sids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    tlids = tlids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    ids = [self.cls] + ids + [self.sep]
                    sids = [0] + sids + [sids[-1]]
                    tlids = [0] + tlids + [tlids[-1]]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'tlids': tlids,
                    'sids': sids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            sids = torch.LongTensor(bundle['sids'])
            tlids = torch.LongTensor(bundle['tlids'])
            return ids, sids, tlids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            sids = torch.LongTensor(bundle['sids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            tlids = torch.LongTensor(bundle['tlids'])
            return ids, sids, tlids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, sids, tlids, rids = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch], [i[3] for i in batch]
            ctext = [i[4] for i in batch]
            rtext = [i[5] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
            tlids = pad_sequence(tlids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, sids, tlids, rids, ids_mask, rids_mask = to_cuda(ids, sids, tlids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'sids': sids, 
                'tlids': tlids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, sids, tlids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, sids, tlids, rids, rids_mask, label = to_cuda(ids, sids, tlids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'sids': sids, 
                'tlids': tlids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualBM25HNDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.gray_cand_num = args['gray_cand_num']

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_bm25_hn_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_bm25_hard_negative(f'{args["root_dir"]}/data/{args["dataset"]}/train_bm25_gray.txt')
            for item in tqdm(data):
                ctx, res, cands = item['q'], item['r'], item['nr']
                utterances = ctx + [res] + cands
                tokens = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids = tokens[:len(ctx)]
                rids = tokens[len(ctx)]
                crids = tokens[-len(cands):]
                # also append the context utterances as the hard negative candidates
                crids.extend(cids)
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]
                ids = [self.cls] + ids + [self.sep]
                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]
                crids = [i[:(self.args["res_max_len"]-2)] for i in crids]
                crids = [[self.cls] + i + [self.sep] for i in crids]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'crids': crids,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids, rids = bundle['ids'], bundle['rids']
            crids = random.sample(bundle['crids'], self.args['gray_cand_num'])
            ids = torch.LongTensor(ids)
            rids = torch.LongTensor(rids)
            crids = [torch.LongTensor(i) for i in crids]
            return ids, rids, crids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = [i[0] for i in batch]
            rids = [i[1] for i in batch]
            crids = []
            for i in batch:
                crids.extend(i[2])
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            crids = pad_sequence(crids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            crids_mask = generate_mask(crids)
            ids, rids, crids, ids_mask, rids_mask, crids_mask = to_cuda(ids, rids, crids, ids_mask, rids_mask, crids_mask)
            return {
                'ids': ids, 
                'rids': rids,
                'hrids': crids,
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'hrids_mask': crids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualDAFullDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_da_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.vocab, self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])

            stop_word_path = f'{args["root_dir"]}/data/stop_word.txt'
            stop_words = load_stop_words(stop_word_path)

            # data augumentation
            vocab = []
            for _, sentences in tqdm(data[:10000]):
                string = ' '.join(sentences)
                words = [w for w in list(jieba.cut(string)) if len(w) <= 2]
                vocab.extend(words)
            self.vocab_ = list(set(vocab))
            print(f'[!] collect {len(vocab)} words for data augmentation')
            # set random seed
            random.seed(args['seed'])
            dataset = []
            for label, sentences in tqdm(data):
                if label == 0:
                    continue
                ctx_ext = da(
                    sentences[:-1],
                    alpha_ss=self.args['alpha_sentence_swap'],
                    alpha_sr=self.args['alpha_synonym_replacement'],
                    alpha_ri=self.args['alpha_random_insertion'],
                    alpha_rs=self.args['alpha_random_swap'],
                    p_rd=self.args['alpha_random_deletion'],
                    num_aug=self.args['aug_num'],
                    lang=self.args['lang'],
                    stop_words=stop_words,
                    vocab=self.vocab_,
                )
                dataset.append(sentences)
                for ctx_ in ctx_ext:
                    dataset.append(ctx_ + [sentences[-1]])

            print(f'[!] collect {len(dataset)} after data augmentation')

            for utterances in tqdm(dataset):
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            # data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save((self.vocab, self.data), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ctext = [i[2] for i in batch]
            rtext = [i[3] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualCTXHNFullDataset(Dataset):

    '''more positive pairs to train the dual bert model'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_ctx_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]
                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            # data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def delete_operator(self, ids):
        # 主要删除 last utterance in multi-turn conversation context
        turn_num = ids.count(self.sep)
        tokens = ids[1:-1]
        valid_tokens = [i for i in ids if i not in [self.sep, self.cls]]
        delete_num = int(len(valid_tokens) * self.args['delete_p'])
        if turn_num > 1:
            # mainly delete the tokens in last utterance
            while True:
                if tokens[-1] != self.sep:
                    tokens.pop()
                    delete_num -= 1
                else:
                    # delete the last [SEP]
                    tokens.pop()
                    break
        # delete
        if delete_num > 0:
            random_del_idx = set(random.sample([idx for idx, i in enumerate(tokens) if i not in [self.sep, self.cls]], delete_num))
            tokens = [i for idx, i in enumerate(tokens) if idx not in random_del_idx]
        ids = [self.cls] + tokens + [self.sep]
        return ids

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            hcids = self.delete_operator(deepcopy(bundle['ids']))
            hcids = torch.LongTensor(hcids)
            return ids, hcids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, hcids, rids = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
            ctext = [i[3] for i in batch]
            rtext = [i[4] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            hcids = pad_sequence(hcids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            hcids_mask = generate_mask(hcids)
            rids_mask = generate_mask(rids)
            ids, hcids, rids, ids_mask, hcids_mask, rids_mask = to_cuda(ids, hcids, rids, ids_mask, hcids_mask, rids_mask)
            return {
                'ids': ids, 
                'hcids': hcids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'hcids_mask': hcids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualSpeakerFullDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_speaker_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                sids, ids = [], []
                for u in cids:
                    ids.extend(u + [self.sep])
                for u in cids[::-2]:
                    sids.extend(u + [self.sep])
                sids.pop()
                ids.pop()
                ids = ids[-(self.args['max_len']-2):] 
                sids = sids[-(self.args['max_len']-2):]
                rids = rids[:(self.args['res_max_len']-2)]
                ids = [self.cls] + ids + [self.sep]
                sids = [self.cls] + sids + [self.sep]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'sids': sids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            # data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    cids, rids_ = item[:-1], item[-1]
                    ids, sids = [], []
                    for u in cids:
                        ids.extend(u + [self.sep])
                    for u in cids[::-2]:
                        sids.extend(u + [self.sep])
                    ids.pop()
                    sids.pop()
                    ids = ids[-(self.args['max_len']-2):] 
                    sids = sids[-(self.args['max_len']-2):]
                    rids_ = rids_[:(self.args['res_max_len']-2)]
                    ids = [self.cls] + ids + [self.sep]
                    sids = [self.cls] + sids + [self.sep]
                    rids_ = [self.cls] + rids_ + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'sids': sids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            sids = torch.LongTensor(bundle['sids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, sids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            sids = torch.LongTensor(bundle['sids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, sids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, sids, rids = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
            ctext = [i[3] for i in batch]
            rtext = [i[4] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            sids = pad_sequence(sids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            sids_mask = generate_mask(sids)
            rids_mask = generate_mask(rids)
            ids, sids, rids, ids_mask, sids_mask, rids_mask = to_cuda(ids, sids, rids, ids_mask, sids_mask, rids_mask)
            return {
                'ids': ids, 
                'sids': sids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'sids_mask': sids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, sids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, sids, rids, rids_mask, label = to_cuda(ids, sids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'sids': sids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullShuffleCtxDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_shuffle_ctx_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            # data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            data = read_text_data_utterances(path, lang=self.args['lang'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]
                self.data.append({
                    'ids': cids,
                    'rids': rids,
                    'ctext': ' [SEP] '.join(utterances[:-1]),
                    'rtext': utterances[-1],
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            # data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids = bundle['ids']
            hcids = deepcopy(cids)
            pcids = deepcopy(cids)
            # ground-truth
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
            ids = [self.cls] + ids + [self.sep]
            # positive
            ids = []
            ratio = random.random()
            if ratio > 0.5:
                pcids = sentence_shuffle(pcids)
            else:
                pcids = sentence_deletion(pcids) 
            pcids = token_shuffle(pcids)

            hcids = sentence_shuffle(hcids)
            token_shuffle
            
            # negative
            hcids = replace_last_utterance(hcids, self.data)
            hcids = token_shuffle(hcids)
            nids = []
            for u in hcids:
                nids.extend(u + [self.sep])
            nids.pop()
            nids = nids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
            nids = [self.cls] + nids + [self.sep]
            # 
            ids = torch.LongTensor(ids)
            nids = torch.LongTensor(nids)
            rids = torch.LongTensor(bundle['rids'])
            return ids, nids, rids, bundle['ctext'], bundle['rtext']
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, nids, rids = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
            ctext = [i[3] for i in batch]
            rtext = [i[4] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            nids = pad_sequence(nids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            nids_mask = generate_mask(nids)
            rids_mask = generate_mask(rids)
            ids, nids, rids, ids_mask, nids_mask, rids_mask = to_cuda(ids, nids, rids, ids_mask, nids_mask, rids_mask)
            return {
                'ids': ids, 
                'nids': nids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'nids_mask': nids_mask, 
                'rids_mask': rids_mask,
                'ctext': ctext,
                'rtext': rtext,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullCtxCLDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_ctx_cl_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids, rids = item[:-1], item[-1]
                self.data.append({
                    'cids': cids,
                    'rids': rids,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            # data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def _packup(self, cids):
        ids = []
        for u in cids:
            ids.extend(u + [self.sep])
        ids.pop()
        ids = [self.cls] + ids[-(self.args['max_len']-2):] + [self.sep]
        return ids

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            cids = bundle['cids']
            rids = bundle['rids']
            # positive and negative
            # insert random sentence before context for positive
            pos_cids = sentence_shuffle(deepcopy(cids))
            pos_ids = torch.LongTensor(self._packup(pos_cids))
            # replace the last utterance for the negative
            neg_cids = replace_last_utterance(deepcopy(cids), self.data)
            neg_ids = torch.LongTensor(self._packup(neg_cids))
            # 
            ids = torch.LongTensor(self._packup(cids))
            rids = torch.LongTensor([self.cls] + rids[:(self.args['res_max_len']-2)] + [self.sep])
            return ids, pos_ids, neg_ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[3] for i in batch]
            pos_ids, neg_ids = [i[1] for i in batch], [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            pos_ids = pad_sequence(pos_ids, batch_first=True, padding_value=self.pad)
            neg_ids = pad_sequence(neg_ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            pos_ids_mask = generate_mask(pos_ids)
            neg_ids_mask = generate_mask(neg_ids)
            rids_mask = generate_mask(rids)
            ids, pos_ids, neg_ids, rids, ids_mask, pos_ids_mask, neg_ids_mask, rids_mask = to_cuda(ids, pos_ids, neg_ids, rids, ids_mask, pos_ids_mask, neg_ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
                'pos_ids': pos_ids,
                'pos_ids_mask': pos_ids_mask,
                'neg_ids': neg_ids,
                'neg_ids_mask': neg_ids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullWithCtxDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_with_context_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for label, utterances in tqdm(data):
                if label == 0:
                    continue
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                # build context ids
                ids = []
                for u in item[:-1]:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]
                # build response ids
                rids = []
                for u in item[-2:]:
                    rids.extend(u + [self.sep])
                rids.pop()
                rids = [self.cls] + rids[-(self.args['res_max_len']-2):] + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            # data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
                for label, utterances in batch:
                    item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    ids = []
                    for u in item[:-1]:
                        ids.extend(u + [self.sep])
                    ids.pop()
                    ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                    ids = [self.cls] + ids + [self.sep]

                    rids_ = []
                    for u in item[-2:]:
                        rids_.extend(u + [self.sep])
                    rids_.pop()
                    rids_ = [self.cls] + rids_[-(self.args['res_max_len']-2):] + [self.sep]
                    rids.append(rids_)
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
            return {
                'ids': ids, 
                'rids': rids, 
                'ids_mask': ids_mask, 
                'rids_mask': rids_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
class BERTDualFullNHPDataset(Dataset):

    '''NHP: next history prediction'''

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_with_nhp_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full_add_next_history_prediction(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            for _, utterances, responses  in tqdm(data):
                cids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                rids = self.vocab.batch_encode_plus(responses, add_special_tokens=False)['input_ids']
                # context
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]
                # response and response history
                rids_1 = [self.cls] + rids[0][:(self.args['res_max_len']-2)] + [self.sep]
                rids_2 = []
                for u in rids:
                    rids_2.extend(u + [self.sep])
                rids_2.pop()
                rids_2 = rids_2[:(self.args['max_len']-2)]    # ignore [CLS] and [SEP]
                rids_2 = [self.cls] + rids_2 + [self.sep]
                self.data.append({
                    'ids': ids,
                    'rids_1': rids_1,
                    'rids_2': rids_2,
                })
        else:
            data = read_text_data_utterances(path, lang=self.args['lang'])
            # DEBUG for Ubuntu Corpus
            # data = data[:10000]    # 1000 sampels for ubunut
            for i in tqdm(range(0, len(data), 10)):
                batch = data[i:i+10]
                rids = []
                gt_text = []
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
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids,
                    'rids': rids,
                    'text': gt_text,
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            rids_1 = torch.LongTensor(bundle['rids_1'])
            rids_2 = torch.LongTensor(bundle['rids_2'])
            return ids, rids_1, rids_2
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, rids_1, rids_2 = [i[0] for i in batch], [i[1] for i in batch], [i[2] for i in batch]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids_1 = pad_sequence(rids_1, batch_first=True, padding_value=self.pad)
            rids_2 = pad_sequence(rids_2, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_1_mask = generate_mask(rids_1)
            rids_2_mask = generate_mask(rids_2)
            ids, rids_1, rids_2, ids_mask, rids_1_mask, rids_2_mask = to_cuda(ids, rids_1, rids_2, ids_mask, rids_1_mask, rids_2_mask)
            return {
                'ids': ids, 
                'rids_1': rids_1, 
                'rids_2': rids_2, 
                'ids_mask': ids_mask, 
                'rids_1_mask': rids_1_mask,
                'rids_2_mask': rids_2_mask,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, text = batch[0]
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'text': text
            }

            
