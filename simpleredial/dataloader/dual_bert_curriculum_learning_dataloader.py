from header import *
from .utils import *
from .util_func import *
from .augmentation import *


class BERTDualCurriculumLearningFullDataset(Dataset):

    def __init__(self, vocab, path, **args):
        # hard for the 2nd trainig stage: hard negative mining
        self.mode = 'easy'
        self.args = args
        self.vocab = vocab
        self.vocab.add_tokens(['[EOS]'])
        self.topk = args['gray_cand_num']

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.splitext(path)[0]}_dual_full_cl_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None

        self.data = []
        if self.args['mode'] == 'train':
            data = read_text_data_utterances_full(path, lang=self.args['lang'], turn_length=self.args['full_turn_length'])
            data = read_bm25_hard_negative(f'{args["root_dir"]}/data/{args["dataset"]}/train_bm25_gray.txt')
            for item in tqdm(data):
                utterances = item['q'] + [item['r']] + item['nr']
                items = self.vocab.batch_encode_plus(utterances, add_special_tokens=False)['input_ids']
                cids = items[:len(item['q'])]
                rids = items[len(item['q'])]
                nrids = items[-len(item['nr']):]
                ids = []
                for u in cids:
                    ids.extend(u + [self.sep])
                ids.pop()
                ids = ids[-(self.args['max_len']-2):]    # ignore [CLS] and [SEP]
                ids = [self.cls] + ids + [self.sep]
                rids = rids[:(self.args['res_max_len']-2)]
                rids = [self.cls] + rids + [self.sep]
                nrids = [[self.cls] + i[:(self.args['res_max_len']-2)] + [self.sep] for i in nrids]
                self.data.append({
                    'ids': ids,
                    'rids': rids,
                    'nrids': nrids,
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
            ids = torch.LongTensor(bundle['ids'])
            rids = torch.LongTensor(bundle['rids'])
            if self.mode == 'easy':
                return ids, rids
            else:
                nrids = random.sample(bundle['nrids'], self.topk)
                nrids = [torch.LongTensor(i) for i in nrids]
                rids = [rids] + nrids
                return ids, rids
        else:
            ids = torch.LongTensor(bundle['ids'])
            rids = [torch.LongTensor(i) for i in bundle['rids']]
            return ids, rids, bundle['label'], bundle['text']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            if self.mode == 'easy':
                ids, rids = [i[0] for i in batch], [i[1] for i in batch]
            else:
                ids, rids = [i[0] for i in batch], []
                for b in batch:
                    rids.extend(b[1])
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
