from header import *
from .utils import *
from .util_func import *


class GPT2Dataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        
        if self.args['mode'] == 'test':
            # for test batch generation
            print(f'[!] set the padding side as the left')
            self.vocab.padding_side = 'left'

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_gpt2_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        random.seed(args['seed'])

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_line_by_line(path)
            self.data = []
            for text in tqdm(data):
                item = self.vocab.encode(text, add_special_tokens=False)
                for idx in range(0, len(item), self.args['max_len']-2):
                    ids = item[idx:idx+self.args['max_len']-2]
                    if len(ids) < self.args['min_len']:
                        continue
                    ids = [self.cls] + ids + [self.sep]
                    self.data.append({'ids': ids})
        else:
            path = f'{args["root_dir"]}/data/{args["dataset"]}/test_gray_simcse.pt'
            data = torch.load(path)
            # random sample 100 samples
            data = random.sample(data, 10)
            self.data = []
            for item in tqdm(data):
                context, pos, neg_responses = item['context'], item['pos_response'], item['neg_responses']
                for neg in neg_responses:
                    # prefix
                    item = self.vocab.encode(context, add_special_tokens=False)
                    ids = [self.cls] + item[-(self.args['max_len']-1):]
                    item = self.vocab.encode(context+pos, add_special_tokens=False)
                    pos_ids = [self.cls] + item[:self.args['max_len']-2] + [self.sep]
                    item = self.vocab.encode(context+neg, add_special_tokens=False)
                    neg_ids = [self.cls] + item[:self.args['max_len']-2] + [self.sep]
                    self.data.append({
                        'ids': ids,
                        'pos_ids': pos_ids,
                        'pos_text': context+pos,
                        'neg_ids': neg_ids,
                        'neg_text': context+neg,
                        'text': context,
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            return ids
        else:
            ids = torch.LongTensor(bundle['ids'])
            pos_ids = torch.LongTensor(bundle['pos_ids'])
            neg_ids = torch.LongTensor(bundle['neg_ids'])
            return ids, pos_ids, neg_ids, bundle['pos_text'], bundle['neg_text'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            return {
                'ids': ids, 
                'mask': mask, 
            }
        else:
            ids = [i[0] for i in batch]
            pos_ids = [i[1] for i in batch]
            neg_ids = [i[2] for i in batch]
            pos_text = [i[3] for i in batch]
            neg_text = [i[4] for i in batch]
            text = [i[5] for i in batch]

            # pad from the left side, batch first
            max_length = max([len(i) for i in ids])
            n_ids = []
            for i in ids:
                ids_ = torch.cat([torch.LongTensor([self.pad] * (max_length - len(i))), i])
                n_ids.append(ids_)
            ids = torch.stack(n_ids)
            mask = generate_mask(ids)
            
            pos_ids = pad_sequence(pos_ids, batch_first=True, padding_value=self.pad)
            pos_ids_mask = generate_mask(pos_ids)
            neg_ids = pad_sequence(neg_ids, batch_first=True, padding_value=self.pad)
            neg_ids_mask = generate_mask(neg_ids)
            ids, mask, pos_ids, pos_ids_mask, neg_ids, neg_ids_mask = to_cuda(ids, mask, pos_ids, pos_ids_mask, neg_ids, neg_ids_mask)
            return {
                'ids': ids, 
                'mask': mask, 
                'pos_ids': pos_ids, 
                'pos_ids_mask': pos_ids_mask, 
                'neg_ids': neg_ids, 
                'neg_ids_mask': neg_ids_mask, 
                'pos_text': pos_text,
                'text': text,
                'neg_text': neg_text,
            }

            
class GPT2UnlikelyhoodDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        
        random.seed(args['seed'])
        
        if self.args['mode'] == 'test':
            # for test batch generation
            print(f'[!] set the padding side as the left')
            self.vocab.padding_side = 'left'

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_gpt2_unlikelyhood_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_unlikelyhood(path)

            # for debug
            data = random.sample(data, 1000)

            self.data = []
            for utterances in tqdm(data):
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                ids, cands, counter = [], [], 0
                for utterance in item:
                    if counter + len(utterance) + 2 > self.args['train_min_len'] and len(cands) > 0:
                        ids = list(chain(*ids))
                        self.data.append({
                            'cids': ids,
                            'pos_rids': utterance,
                            'cands': cands,
                        })
                        ids, cands = [], []
                    else:
                        ids.append(utterance)
                        cands.append(utterance)
                        counter += len(utterance)
        else:
            path = f'{args["root_dir"]}/data/{args["dataset"]}/test_gray_simcse.pt'
            data = torch.load(path)
            # random sample 100 samples
            data = random.sample(data, 10)
            self.data = []
            for item in tqdm(data):
                context, pos, neg_responses = item['context'], item['pos_response'], item['neg_responses']
                # prefix
                item = self.vocab.encode(context, add_special_tokens=False)
                ids = [self.cls] + item[-(self.args['max_len']-1):]
                cids, rids = self.vocab.batch_encode_plus([context, pos], add_special_tokens=False)['input_ids']
                self.truncate_pair(cids, rids, self.args['max_len'])
                pos_ids = [self.cls] + cids + rids + [self.sep]
                pos_label = [0] * (len(cids) + 1) + rids + [self.sep]
                neg_ids_total, neg_ids_label_total, neg_text_total = [], [], []
                for neg in neg_responses:
                    cids, rids = self.vocab.batch_encode_plus([context, neg], add_special_tokens=False)['input_ids']
                    self.truncate_pair(cids, rids, self.args['max_len'])
                    neg_ids = [self.cls] + cids + rids + [self.sep]
                    neg_label = [0] * (len(cids) + 1) + rids + [self.sep]
                    neg_ids_total.append(neg_ids)
                    neg_ids_label_total.append(neg_label)
                    neg_text_total.append(context+neg)
                self.data.append({
                    'ids': ids,
                    'pos_ids': pos_ids,
                    'pos_label': pos_label,
                    'pos_text': context+pos,
                    'neg_ids': neg_ids_total,
                    'neg_label': neg_ids_label_total,
                    'neg_text': neg_text_total,
                    'text': context,
                })

    def __len__(self):
        return len(self.data)

    def truncate_pair(self, ids, rids, max_len):
        max_len -= 2
        while True:
            l = len(ids) + len(rids)
            if l <= max_len:
                break
            if len(ids) > len(rids):
                ids.pop(0)
            else:
                rids.pop()

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids, pos_rids, cands = deepcopy(bundle['cids']), deepcopy(bundle['pos_rids']), deepcopy(bundle['cands'])
            cand = random.choice(cands)
            neg_ids = deepcopy(ids)

            truncate_pair(ids, pos_rids, self.args['train_max_len'])
            gpt2_ids = [self.cls] + ids + pos_rids + [self.sep]
            bert_label = [0] * (len(ids) + 1) + pos_rids + [self.sep]

            truncate_pair(neg_ids, cand, self.args['train_max_len'])
            neg_gpt2_ids = [self.cls] + neg_ids + cand + [self.sep]
            neg_bert_label = [0] * (len(neg_ids) + 1) + cand + [self.sep]
            return gpt2_ids, bert_label, neg_gpt2_ids, neg_bert_label
        else:
            ids = torch.LongTensor(bundle['ids'])
            pos_ids = torch.LongTensor(bundle['pos_ids'])
            neg_ids = [torch.LongTensor(i) for i in bundle['neg_ids']]
            pos_label = torch.LongTensor(bundle['pos_label'])
            neg_label = [torch.LongTensor(i) for i in bundle['neg_label']]
            return ids, pos_ids, neg_ids, bundle['pos_text'], bundle['neg_text'], bundle['text'], pos_label, neg_label

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            gpt2_ids, bert_label, neg_gpt2_ids, neg_bert_label = [], [], [], []
            for a, b, c, d in batch:
                gpt2_ids.append(torch.LongTensor(a))
                bert_label.append(torch.LongTensor(b))
                neg_gpt2_ids.append(torch.LongTensor(c))
                neg_bert_label.append(torch.LongTensor(d))
            gpt2_ids = pad_sequence(gpt2_ids, batch_first=True, padding_value=self.pad)
            neg_gpt2_ids = pad_sequence(neg_gpt2_ids, batch_first=True, padding_value=self.pad)
            bert_label = pad_sequence(bert_label, batch_first=True, padding_value=self.pad)
            neg_bert_label = pad_sequence(neg_bert_label, batch_first=True, padding_value=self.pad)
            gpt2_mask = generate_mask(gpt2_ids)
            neg_gpt2_mask = generate_mask(neg_gpt2_ids)
            gpt2_ids, gpt2_mask, bert_label = to_cuda(gpt2_ids, gpt2_mask, bert_label)
            neg_gpt2_ids, neg_gpt2_mask, neg_bert_label = to_cuda(neg_gpt2_ids, neg_gpt2_mask, neg_bert_label)
            return {
                'gpt2_ids': gpt2_ids,
                'gpt2_mask': gpt2_mask,
                'bert_label': bert_label,
                'neg_gpt2_ids': neg_gpt2_ids,
                'neg_gpt2_mask': neg_gpt2_mask,
                'neg_bert_label': neg_bert_label,
            }
        else:
            neg_ids_, neg_text_, neg_ids_mask_, neg_label_ = [], [], [], []
            for i in range(10):
                neg_ids = [j[2][i] for j in batch]
                neg_text = [j[4][i] for j in batch]
                neg_label = [j[7][i] for j in batch]
                neg_ids = pad_sequence(neg_ids, batch_first=True, padding_value=self.pad)
                neg_label = pad_sequence(neg_label, batch_first=True, padding_value=self.pad)
                neg_ids_mask = generate_mask(neg_ids)
                neg_ids, neg_ids_mask, neg_label = to_cuda(neg_ids, neg_ids_mask, neg_label)

                neg_ids_.append(neg_ids)
                neg_ids_mask_.append(neg_ids_mask)
                neg_label_.append(neg_label)
                neg_text_.append(neg_text)

            ids = [i[0] for i in batch]
            pos_ids = [i[1] for i in batch]
            pos_text = [i[3] for i in batch]
            text = [i[5] for i in batch]
            pos_label = [i[6] for i in batch]

            # pad from the left side, batch first
            max_length = max([len(i) for i in ids])
            n_ids = []
            for i in ids:
                ids_ = torch.cat([torch.LongTensor([self.pad] * (max_length - len(i))), i])
                n_ids.append(ids_)
            ids = torch.stack(n_ids)
            mask = generate_mask(ids)
            
            pos_ids = pad_sequence(pos_ids, batch_first=True, padding_value=self.pad)
            pos_label = pad_sequence(pos_label, batch_first=True, padding_value=self.pad)
            pos_ids_mask = generate_mask(pos_ids)
            ids, mask, pos_ids, pos_ids_mask, pos_label = to_cuda(ids, mask, pos_ids, pos_ids_mask, pos_label)
            return {
                'ids': ids, 
                'mask': mask, 
                'pos_ids': pos_ids, 
                'pos_label': pos_label, 
                'pos_ids_mask': pos_ids_mask, 
                'neg_ids': neg_ids_, 
                'neg_label': neg_label_, 
                'neg_ids_mask': neg_ids_mask_, 
                'pos_text': pos_text,
                'text': text,
                'neg_text': neg_text_,
            }

            
class GPT2WithNegDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        
        if self.args['mode'] == 'test':
            # for test batch generation
            print(f'[!] set the padding side as the left')
            self.vocab.padding_side = 'left'

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_gpt2_with_neg_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        random.seed(args['seed'])

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_line_by_line(path)
            self.data = []
            for text in tqdm(data):
                item = self.vocab.encode(text, add_special_tokens=False)
                for idx in range(0, len(item), self.args['max_len']-2):
                    ids = item[idx:idx+self.args['max_len']-2]
                    if len(ids) < self.args['min_len']:
                        continue
                    ids = [self.cls] + ids + [self.sep]
                    self.data.append({'ids': ids})
        else:
            path = f'{args["root_dir"]}/data/{args["dataset"]}/test_gray_simcse.pt'
            data = torch.load(path)
            # random sample 100 samples
            data = random.sample(data, 10)
            self.data = []
            for item in tqdm(data):
                context, pos, neg_responses = item['context'], item['pos_response'], item['neg_responses']
                for neg in neg_responses:
                    # prefix
                    item = self.vocab.encode(context, add_special_tokens=False)
                    ids = [self.cls] + item[-(self.args['max_len']-1):]
                    item = self.vocab.encode(context+pos, add_special_tokens=False)
                    pos_ids = [self.cls] + item[:self.args['max_len']-2] + [self.sep]
                    item = self.vocab.encode(context+neg, add_special_tokens=False)
                    neg_ids = [self.cls] + item[:self.args['max_len']-2] + [self.sep]
                    self.data.append({
                        'ids': ids,
                        'pos_ids': pos_ids,
                        'pos_text': context+pos,
                        'neg_ids': neg_ids,
                        'neg_text': context+neg,
                        'text': context,
                    })

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = torch.LongTensor(bundle['ids'])
            return ids
        else:
            ids = torch.LongTensor(bundle['ids'])
            pos_ids = torch.LongTensor(bundle['pos_ids'])
            neg_ids = torch.LongTensor(bundle['neg_ids'])
            return ids, pos_ids, neg_ids, bundle['pos_text'], bundle['neg_text'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            return {
                'ids': ids, 
                'mask': mask, 
            }
        else:
            ids = [i[0] for i in batch]
            pos_ids = [i[1] for i in batch]
            neg_ids = [i[2] for i in batch]
            pos_text = [i[3] for i in batch]
            neg_text = [i[4] for i in batch]
            text = [i[5] for i in batch]

            # pad from the left side, batch first
            max_length = max([len(i) for i in ids])
            n_ids = []
            for i in ids:
                ids_ = torch.cat([torch.LongTensor([self.pad] * (max_length - len(i))), i])
                n_ids.append(ids_)
            ids = torch.stack(n_ids)
            mask = generate_mask(ids)
            
            pos_ids = pad_sequence(pos_ids, batch_first=True, padding_value=self.pad)
            pos_ids_mask = generate_mask(pos_ids)
            neg_ids = pad_sequence(neg_ids, batch_first=True, padding_value=self.pad)
            neg_ids_mask = generate_mask(neg_ids)
            ids, mask, pos_ids, pos_ids_mask, neg_ids, neg_ids_mask = to_cuda(ids, mask, pos_ids, pos_ids_mask, neg_ids, neg_ids_mask)
            return {
                'ids': ids, 
                'mask': mask, 
                'pos_ids': pos_ids, 
                'pos_ids_mask': pos_ids_mask, 
                'neg_ids': neg_ids, 
                'neg_ids_mask': neg_ids_mask, 
                'pos_text': pos_text,
                'text': text,
                'neg_text': neg_text,
            }
