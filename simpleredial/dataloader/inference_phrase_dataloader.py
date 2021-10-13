from header import *
from .utils import *
from .util_func import *

class BERTDualInferencePhraseDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_phrase_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # except the response in the train dataset, test dataset responses are included for inference test
        # for inference[gray mode] do not use the test set responses
        data = read_text_data_utterances(path, lang=self.args['lang'])
        self.data = []
        for label, utterances in tqdm(data):
            if label == 0:
                continue
            item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
            ids = []
            for u in item:
                ids.extend(u+[self.sep])
            ids.pop()
            ids = ids[-self.args['max_len']-2:]
            ids = [self.cls] + ids + [self.sep]
            self.data.append({
                'ids': ids, 
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        return ids

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = pad_sequence(batch, batch_first=True, padding_value=self.pad)
        ids_mask = generate_mask(ids)
        ids, ids_mask = to_cuda(ids, ids_mask)
        return {
            'ids': ids, 
            'mask': ids_mask, 
        }
