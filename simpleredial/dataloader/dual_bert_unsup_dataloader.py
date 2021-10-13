from header import *
from .utils import *
from .util_func import *


class BERTDualUnsupervisedDataset(Dataset):

    '''only for ext_douban dataset'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_unsup_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        path = f'{args["root_dir"]}/data/ext_douban/train.txt'
        data = read_extended_douban_corpus(path)

        self.data = []
        inner_bsz = 256
        for idx in tqdm(range(0, len(data), inner_bsz)):
            utterances = data[idx:idx+inner_bsz]
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = []
            for u in item:
                u = u[:(self.args['res_max_len']-2)]
                u = [self.cls] + u + [self.sep]
                ids.append(u)
            self.data.extend([{
                    'ids': i,
                    'text': j,
                } for i, j in zip(ids, utterances)
            ])
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        return ids, bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids, text = [i[0] for i in batch], [i[1] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'ids_mask': ids_mask, 
            'text': text,
        }
