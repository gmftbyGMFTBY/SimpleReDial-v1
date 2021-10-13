from header import *
from .utils import *
from .util_func import *

class BERTDualInferenceContextDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_ctx_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = []
            for u in item[:-1]:
                ids.extend(u + [self.eos])
            ids.pop()
            ids = ids[-self.args['max_len']+2:]
            ids = [self.cls] + ids + [self.sep]
            self.data.append({
                'ids': ids, 
                'context': utterances[:-1],
                'response': utterances[-1],
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        context = bundle['context']
        response = bundle['response']
        return ids, context, response

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        context = [i[1] for i in batch]
        response = [i[2] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'mask': ids_mask, 
            'context': context,
            'response': response
        }

        
class BERTDualInferenceFullContextDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_full_ctx_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=args['full_turn_length'])
        self.data = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = []
            for u in item[:-1]:
                ids.extend(u + [self.eos])
            ids.pop()
            ids = ids[-self.args['max_len']+2:]
            ids = [self.cls] + ids + [self.sep]
            self.data.append({
                'ids': ids, 
                'context': utterances[:-1],
                'response': utterances[-1],
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        context = bundle['context']
        response = bundle['response']
        return ids, context, response

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        context = [i[1] for i in batch]
        response = [i[2] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'mask': ids_mask, 
            'context': context,
            'response': response
        }

        
class BERTDualInferenceFullContextSingleExtendDataset(Dataset):
    
    '''each in-dataset utterance will be treated as the extended context for training'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_full_ctx_ext_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        data = read_response_data_full(path, lang=self.args['lang'], turn_length=5)
        self.data = []
        for utterance in tqdm(data):
            ids = self.vocab.encode(utterance, add_special_tokens=False)
            ids = ids[-self.args['max_len']+2:]
            ids = [self.cls] + ids + [self.sep]
            self.data.append({
                'ids': ids, 
                'context': utterance, 
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        context = bundle['context']
        return ids, context

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        context = [i[1] for i in batch]
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'mask': ids_mask, 
            'context': context,
            'response': context,
        }
