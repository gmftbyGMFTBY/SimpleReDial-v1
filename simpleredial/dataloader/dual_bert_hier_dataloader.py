from header import *
from .utils import *
from .util_func import *
from .augmentation import *


class BERTDualFullHierDataset(Dataset):

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
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_hier_{suffix}.pt'
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
                ids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids]
                self.data.append({
                    'ids': ids[:-1],
                    'rids': ids[-1],
                    'turn_length': len(ids) - 1,
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
                    ids = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                    ids = [[self.cls] + i[-(self.args['max_len']-2):] + [self.sep] for i in ids]
                    rids.append(ids[-1])
                    if label == 1:
                        gt_text.append(utterances[-1])
                self.data.append({
                    'label': [b[0] for b in batch],
                    'ids': ids[:-1],
                    'rids': rids,
                    'text': gt_text,
                    'turn_length': len(ids) - 1
                })    
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        if self.args['mode'] == 'train':
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            rids = torch.LongTensor(bundle['rids'])
            return ids, rids, bundle['turn_length']
        else:
            ids = [torch.LongTensor(i) for i in bundle['ids']]
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['turn_length']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = []
            for i in batch:
                ids.extend(i[0])
            rids = [i[1] for i in batch]
            turn_length = [i[2] for i in batch]
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
                'turn_length': turn_length,
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids, rids, label, turn_length = batch[0]
            turn_length = [turn_length]
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
            rids_mask = generate_mask(rids)
            label = torch.LongTensor(label)
            ids, rids, rids_mask, label = to_cuda(ids, rids, rids_mask, label)
            return {
                'ids': ids, 
                'rids': rids, 
                'rids_mask': rids_mask, 
                'label': label,
                'turn_length': turn_length
            }
