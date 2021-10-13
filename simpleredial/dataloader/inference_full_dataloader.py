from header import *
from .utils import *
from .randomaccess import *

class BERTDualFullWithSourceInferenceDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.path = f'{args["root_dir"]}/data/{args["dataset"]}/inference.txt'
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/inference.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccessReader Object over')
        else:
            self.reader = RandomAccessReader(self.path)
            self.reader.init()
            torch.save(self.reader, rar_path)
            print(f'[!] save the random access reader file into {rar_path}')
        self.reader.init_file_handler()
        self.size = self.reader.size
        print(f'[!] dataset size: {self.size}')
                
    def _length_limit(self, ids):
        # only inference the responses
        if len(ids) > self.args['max_len']:
            ids = ids[:self.args['max_len']:]
        return ids
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        line = self.reader.get_line(i).strip().split('\t')[-1]
        line = json.loads(line)
        ids = self.vocab.encode(line[0])
        ids = self._length_limit(ids)
        ids = torch.LongTensor(ids)
        # text, title, url
        return ids, line[0], line[1], line[2]

    def save(self):
        pass
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        title_text = [i[2] for i in batch]
        url_text = [i[3] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = self.generate_mask(rid)
        if torch.cuda.is_available():
            rid, rid_mask = rid.cuda(), rid_mask.cuda()
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text,
            'title': title_text,
            'url': url_text,
        }
        

class BERTDualFullInferenceDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.path = f'{args["root_dir"]}/data/{args["dataset"]}/inference.txt'
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/inference.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccessReader Object over')
        else:
            self.reader = RandomAccessReader(self.path)
            self.reader.init()
            torch.save(self.reader, rar_path)
            print(f'[!] save the random access reader file into {rar_path}')
        self.reader.init_file_handler()
        self.size = self.reader.size
        print(f'[!] dataset size: {self.size}')
                
    def _length_limit(self, ids):
        # only inference the responses
        if len(ids) > self.args['max_len']:
            ids = ids[:self.args['max_len']:]
        return ids
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        line = self.reader.get_line(i).strip()
        ids = self.vocab.encode(line)
        ids = self._length_limit(ids)
        ids = torch.LongTensor(ids)
        return ids, line

    def save(self):
        pass
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = self.generate_mask(rid)
        if torch.cuda.is_available():
            rid, rid_mask = rid.cuda(), rid_mask.cuda()
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text
        }

        
class BERTDualFullCLInferenceDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.path = f'{args["root_dir"]}/data/{args["dataset"]}/inference.txt'
        rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/inference.rar'
        if os.path.exists(rar_path):
            self.reader = torch.load(rar_path)
            print(f'[!] load RandomAccessReader Object over')
        else:
            self.reader = RandomAccessReader(self.path)
            self.reader.init()
            torch.save(self.reader, rar_path)
            print(f'[!] save the random access reader file into {rar_path}')
        self.reader.init_file_handler()
        self.size = self.reader.size
        print(f'[!] dataset size: {self.size}')
                
    def _length_limit(self, ids):
        # only inference the responses
        if len(ids) > self.args['max_len']:
            ids = ids[:self.args['max_len']:]
        return ids
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        line = self.reader.get_line(i).strip()
        ids = self.vocab.encode(line)
        ids = self._length_limit(ids)
        ids = torch.LongTensor(ids)
        return ids, line

    def save(self):
        pass
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = self.generate_mask(rid)
        if torch.cuda.is_available():
            rid, rid_mask = rid.cuda(), rid_mask.cuda()
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text
        }
