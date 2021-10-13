from header import *
from .utils import *
from .util_func import *

class BERTDualFullFilterInferenceDataset(Dataset):

    '''Only for full-rank, which only the response in the train.txt is used for inference'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_full_filter_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
        self.data = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            cids, rids = item[:-1], item[-1]
            ids = []
            for u in cids:
                ids.extend(u + [self.sep])
            ids.pop()
            ids = ids[-(self.args['ctx_max_len']-2):]
            rids = rids[:self.args['max_len']-2]
            ids = [self.cls] + ids + [self.sep]
            rids = [self.cls] + rids + [self.sep]
            self.data.append({
                'ids': ids, 
                'rids': rids,
                'ctext': utterances[:-1],
                'rtext': utterances[-1]
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        rids = torch.LongTensor(bundle['rids'])
        return ids, rids, bundle['ctext'], bundle['rtext']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        rids = [i[1] for i in batch]
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
            'rids_mask': rids_mask, 
            'ids_mask': ids_mask, 
            'ctext': ctext,
            'rtext': rtext,
        }
