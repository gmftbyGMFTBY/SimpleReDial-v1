from header import *
from .utils import *
from .util_func import *

class BERTDualPTDataset(Dataset):

    '''Dual bert dataloader for post train (MLM)'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')
        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk])

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_post_train_{suffix}.pt'
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
                ids = ids[-(self.args['max_len']-2):]
                ids = [self.cls] + ids + [self.sep]
                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]
                num_valid_ctx = len([i for i in ids if i not in self.special_tokens])
                num_valid_res = len([i for i in rids if i not in self.special_tokens])
                if num_valid_ctx < self.args['min_ctx_len'] or num_valid_res < self.args['min_res_len']:
                    continue
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
            o_ids, o_rids = bundle['ids'], bundle['rids']
            ids, rids = deepcopy(o_ids), deepcopy(o_rids)
            mask_labels_ids = mask_sentence(
                ids,
                self.args['min_mask_num'],
                self.args['max_mask_num'],
                self.args['masked_lm_prob'],
                special_tokens=self.special_tokens,
                mask=self.mask,
                vocab_size=len(self.vocab)
            )
            mask_labels_rids = mask_sentence(
                rids,
                self.args['min_mask_num'],
                self.args['max_mask_num'],
                self.args['masked_lm_prob'],
                special_tokens=self.special_tokens,
                mask=self.mask,
                vocab_size=len(self.vocab)
            )
            return ids, mask_labels_ids, rids, mask_labels_rids, o_ids, o_rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; dataset size: {len(self.data)}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids, mask_labels_ids, rids, mask_labels_rids, o_ids, o_rids = [], [], [], [], [], []
            for a, b, c, d, e, f in batch:
                ids.append(torch.LongTensor(a))
                mask_labels_ids.append(torch.LongTensor(b))
                rids.append(torch.LongTensor(c))
                mask_labels_rids.append(torch.LongTensor(d))
                o_ids.append(torch.LongTensor(e))
                o_rids.append(torch.LongTensor(f))
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            o_ids = pad_sequence(o_ids, batch_first=True, padding_value=self.pad)
            o_rids = pad_sequence(o_rids, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            rids_mask = generate_mask(rids)
            mask_labels_ids = pad_sequence(mask_labels_ids, batch_first=True, padding_value=self.pad)
            mask_labels_rids = pad_sequence(mask_labels_rids, batch_first=True, padding_value=self.pad)
            o_ids, o_rids, ids, ids_mask, mask_labels_ids, rids, rids_mask, mask_labels_rids = to_cuda(o_ids, o_rids, ids, ids_mask, mask_labels_ids, rids, rids_mask, mask_labels_rids)
            return {
                'ids': ids,
                'ids_mask': ids_mask,
                'mask_labels_ids': mask_labels_ids,
                'rids': rids,
                'rids_mask': rids_mask,
                'mask_labels_rids': mask_labels_rids,
                'o_ids': o_ids,
                'o_rids': o_rids,
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
