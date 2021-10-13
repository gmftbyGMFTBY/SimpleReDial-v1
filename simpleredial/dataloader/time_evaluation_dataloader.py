from header import *
from .utils import *
from .util_func import *


class BERTFTTimeDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.candidate_size = args['candidate_size']
        
        data = read_text_data_utterances(path, lang=self.args['lang'])
        train_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
        train_data = read_text_data_utterances(train_path, lang=self.args['lang'])
        self.candidates = list(chain(*[i[1] for i in train_data]))

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_ft_{suffix}.pt'
        
        self.data = []
        for i in tqdm(range(0, len(data), 10)):
            batch = data[i:i+10]
            ids, tids = [], []
            context, responses = [], []
            # ext samples for rating
            ext_sample = random.sample(self.candidates, self.candidate_size-10)
            context = batch[0][1]
            for item in ext_sample:
                item = context + [item]
                batch.append((0, item))
            for b in batch:
                label = b[0]
                utterances = b[1]
                item = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids = []
                for u in item[:-1]:
                    cids.extend(u + [self.eos])
                cids.pop()
                rids = item[-1]
                truncate_pair(cids, rids, self.args['max_len'])
                ids_ = [self.cls] + cids + [self.sep] + rids + [self.sep]
                tids_ = [0] * (len(cids) + 2) + [1] * (len(rids) + 1)
                ids.append(ids_)
                tids.append(tids_)
                responses.append(utterances[-1])
            context = ' [SEP] '.join(utterances[:-1])
            self.data.append({
                'label': [b[0] for b in batch],
                'ids': ids,
                'tids': tids,
                'context': context,
                'responses': responses,
            })    

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = [torch.LongTensor(i) for i in bundle['ids']]
        tids = [torch.LongTensor(i) for i in bundle['tids']]
        context = bundle['context']
        responses = bundle['responses']
        return ids, tids, bundle['label'], context, responses

    def save(self):
        pass
        
    def collate(self, batch):
        # batch size is batch_size * 10
        ids, tids, label = [], [], []
        context, responses = [], []
        for b in batch:
            ids.extend(b[0])
            tids.extend(b[1])
            label.extend(b[2])
            context.append(b[3])
            responses.extend(b[4])
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        label = torch.LongTensor(label)
        ids, tids, mask, label = to_cuda(ids, tids, mask, label)
        return {
            'ids': ids, 
            'tids': tids, 
            'mask': mask, 
            'label': label,
            'context': context,
            'responses': responses,
        }

class BERTDualTimeDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.candidate_size = args['candidate_size']
        
        data = read_text_data_utterances(path, lang=self.args['lang'])
        train_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
        train_data = read_text_data_utterances(train_path, lang=self.args['lang'])
        self.candidates = list(chain(*[i[1] for i in train_data]))

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_{suffix}.pt'

        self.data = []
        for i in tqdm(range(0, len(data), 10)):
            batch = data[i:i+10]
            rids = []
            gt_text = []

            # ext samples for rating
            ext_sample = random.sample(self.candidates, self.candidate_size-10)
            context = batch[0][1]
            for item in ext_sample:
                item = context + [item]
                batch.append((0, item))

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
        ids = torch.LongTensor(bundle['ids'])
        rids = [torch.LongTensor(i) for i in bundle['rids']]
        return ids, rids, bundle['label'], bundle['text']

    def save(self):
        pass
        
    def collate(self, batch):
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
