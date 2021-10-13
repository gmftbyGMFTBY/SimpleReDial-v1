from header import *
from .utils import *
from .util_func import *


class BERTDualPhraseDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_phrase_{suffix}.pt'
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
                ids = []
                for u in item:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]
                self.data.append({
                    'ids': ids,
                    'text': ' [SEP] '.join(utterances),
                })
        else:
            for i in tqdm(range(0, len(data), 10)):
                _, utterances = data[i]
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                ids = []
                for u in item:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]
                self.data.append({
                    'ids': ids,
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
            return ids

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
            ids_mask = generate_mask(ids)
            ids, ids_mask = to_cuda(ids, ids_mask)
            return {
                'ids': ids, 
                'ids_mask': ids_mask, 
            }
        else:
            # batch size is batch_size * 10
            assert len(batch) == 1
            ids = batch[0].unsqueeze(0)
            ids = to_cuda(ids)[0]
            return {
                'ids': ids, 
            }
