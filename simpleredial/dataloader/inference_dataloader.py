from header import *
from .utils import *
from .util_func import *

class BERTDualInferenceDataset(Dataset):

    '''Only for full-rank, which only the response in the train.txt is used for inference'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # ========== for full-rank response recall ========= #
        responses = read_response_data(path, lang=self.args['lang'])
        responses = list(set(responses))
        print(f'[!] load {len(responses)} responses for inference finally')

        self.data = []
        for res in tqdm(responses):
            rids = length_limit_res(self.vocab.encode(res), self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': rids, 
                'text': res
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text
        }


class BERTDualInferenceFullDataset(Dataset):

    '''all the in-dataset response'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_full_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        responses = read_response_data_full(path, lang=self.args['lang'], turn_length=5)
        responses = list(set(responses))
        print(f'[!] load {len(responses)} responses for inference')

        self.data = []
        for res in tqdm(responses):
            rids = length_limit_res(self.vocab.encode(res), self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': rids, 
                'text': res
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text
        }

        
class BERTDualInferenceFullEXTDataset(Dataset):

    '''all the in-dataset response and out-dataset response'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_full_ext_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        responses = read_response_data_full(path, lang=self.args['lang'], turn_length=5)
        ext_responses = read_extended_douban_corpus(f'{args["root_dir"]}/data/ext_douban/train.txt')
        responses += ext_responses
        responses = list(set(responses))
        print(f'[!] load {len(responses)} responses for inference')

        self.data = []
        for res in tqdm(responses):
            rids = length_limit_res(self.vocab.encode(res), self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': rids, 
                'text': res
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text
        }

        
class BERTDualInferenceFullForOne2ManyDataset(Dataset):

    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_full_for_one2many_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # context, response = read_response_and_context_full(path, lang=self.args['lang'], turn_length=args['full_turn_length'])
        context, response = read_response_and_context_full(path, lang=self.args['lang'], turn_length=1)

        self.data = []
        for ctx, res in tqdm(list(zip(context, response))):
            item = self.vocab.batch_encode_plus(ctx + [res], add_special_tokens=False)['input_ids']
            ids = [self.cls]
            for u in item[:-1]:
                ids.extend(u+[self.sep])
            ids[-1] = self.sep
            ids = length_limit(ids, self.args['ctx_max_len'])
            rids = [self.cls] + item[-1] + [self.sep]
            rids = length_limit_res(rids, self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': ids,
                'rids': rids,
                'ctext': ctx,
                'rtext': res,
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        ids = torch.LongTensor(bundle['ids'])
        rids = torch.LongTensor(bundle['rids'])
        rid_text = bundle['rtext']
        cid_text = bundle['ctext']
        return ids, rids, rid_text, cid_text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        ids = [i[0] for i in batch]
        rids = [i[1] for i in batch]
        rid_text = [i[2] for i in batch]
        cid_text = [i[3] for i in batch]
        rids = pad_sequence(rids, batch_first=True, padding_value=self.pad)
        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        rids_mask = generate_mask(rids)
        ids_mask = generate_mask(ids)
        ids, rids, ids_mask, rids_mask = to_cuda(ids, rids, ids_mask, rids_mask)
        return {
            'ids': ids,
            'rids': rids,
            'ids_mask': ids_mask, 
            'rids_mask': rids_mask, 
            'ctext': cid_text,
            'rtext': rid_text,
        }

        
class BERTDualInferenceWithSourceDataset(Dataset):

    '''Only for full-rank, which only the response in the train.txt is used for inference'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_with_source_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # ========== for full-rank response recall ========= #
        dataset = read_response_data_with_context(path, lang=self.args['lang'])
        self.data = []
        for res in tqdm(dataset):
            ctx = dataset[res]
            rids = length_limit_res(self.vocab.encode(res), self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': rids, 
                'text': res,
                'ctext': ctx,
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text, bundle['ctext']

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        ctext = [i[2] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text,
            'ctext': ctext,
        }

class BERTDualInferenceWithTestDataset(Dataset):

    '''Only for full-rank, which only the response in the train.txt is used for inference'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_with_test_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        # ========== for full-rank response recall ========= #
        responses = read_response_data(path, lang=self.args['lang'])
        responses = list(set(responses))
        # add the test dataset
        test_path = f'{os.path.split(path)[0]}/test.txt'
        test_responses = read_response_data_test(test_path, lang=self.args['lang'])
        test_responses = list(set(test_responses))
        responses += test_responses
        responses = list(set(responses))
        print(f'[!] load {len(responses)} responses for inference finally')

        self.data = []
        for res in tqdm(responses):
            rids = length_limit_res(self.vocab.encode(res), self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': rids, 
                'text': res
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text
        }

class BERTDualInferenceFullWithTestDataset(Dataset):

    '''all the in-dataset response'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_full_with_test_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        responses = read_response_data_full(path, lang=self.args['lang'], turn_length=5)
        # add the test dataset
        test_path = f'{os.path.split(path)[0]}/test.txt'
        test_responses = read_response_data_test(test_path, lang=self.args['lang'])
        test_responses = list(set(test_responses))
        responses += test_responses
        responses = list(set(responses))
        print(f'[!] load {len(responses)} responses for inference finally')

        self.data = []
        for res in tqdm(responses):
            rids = length_limit_res(self.vocab.encode(res), self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': rids, 
                'text': res
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text
        }

        
class BERTDualInferenceFullEXTWithTestDataset(Dataset):

    '''all the in-dataset response and out-dataset response'''
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        suffix = args['tokenizer'].replace('/', '_')
        self.pp_path = f'{os.path.split(path)[0]}/inference_full_ext_with_test_{suffix}.pt'
        if os.path.exists(self.pp_path):
            self.data = torch.load(self.pp_path)
            print(f'[!] load preprocessed file from {self.pp_path}')
            return None
        responses = read_response_data_full(path, lang=self.args['lang'], turn_length=5)
        ext_responses = read_extended_douban_corpus(f'{args["root_dir"]}/data/ext_douban/train.txt')
        responses += ext_responses
        # add the test dataset
        test_path = f'{os.path.split(path)[0]}/test.txt'
        test_responses = read_response_data_test(test_path, lang=self.args['lang'])
        test_responses = list(set(test_responses))
        responses += test_responses
        responses = list(set(responses))
        print(f'[!] load {len(responses)} responses for inference')

        self.data = []
        for res in tqdm(responses):
            rids = length_limit_res(self.vocab.encode(res), self.args['max_len'], sep=self.sep)
            self.data.append({
                'ids': rids, 
                'text': res
            })
                
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        rid = torch.LongTensor(bundle['ids'])
        rid_text = bundle['text']
        return rid, rid_text

    def save(self):
        data = torch.save(self.data, self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}')
        
    def collate(self, batch):
        rid = [i[0] for i in batch]
        rid_text = [i[1] for i in batch]
        rid = pad_sequence(rid, batch_first=True, padding_value=self.pad)
        rid_mask = generate_mask(rid)
        rid, rid_mask = to_cuda(rid, rid_mask)
        return {
            'ids': rid, 
            'mask': rid_mask, 
            'text': rid_text
        }
