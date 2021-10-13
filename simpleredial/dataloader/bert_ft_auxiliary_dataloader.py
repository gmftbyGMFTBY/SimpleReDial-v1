from header import *
from .utils import *
from .util_func import *


class BERTFTAuxiliaryDataset(Dataset):

    '''Bert-ft auxiliary dataset for essay evaluation'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')

        self.mapping = {
            0: 'deletion',        
            1: 'insertion',        
            2: 'replace',        
            3: 'swap',        
            4: 'mlm',        
            5: 'clm',        
        }
        self.mapping_id = 0

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        if self.args['mode'] == 'train':
            data, sentences = read_essay_dataset(path)
            self.sentences = sentences
            context_length = 5
            self.data = []
            for passage in data:
                items = self.vocab.batch_encode_plus(essay, add_special_tokens=False)['intpu_ids']
                self.data.append(items)
        else:
            pass

    def __len__(self):
        return len(self.data)

    def packup_deletion(self, sentences, idx=-1):
        ids = []
        for u in sentences:
            if idx in range(len(sentences)):
                pass
            else:
                ids.extend(u + [self.sep])
        ids.pop()
        ids = ids[:self.args['max_len']-2]
        ids = [self.cls] + ids + [self.sep]
        return ids

    def __getitem__(self, i):
        passage = self.data[i]
        if self.args['mode'] == 'train':
            if self.mapping_id == 0:
                # deletion
                ids, labels = self.packup_deletion(passage)
            
        self.mapping_id = 0 if self.mapping_id + 1 > 5 else self.mapping_id + 1
        return ids, labels

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        pass

    def packup_deletion(self, passage):
        pass
