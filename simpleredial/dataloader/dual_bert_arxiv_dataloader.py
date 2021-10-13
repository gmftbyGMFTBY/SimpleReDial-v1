from header import *
from .utils import *
from .randomaccess import *


class BERTDualArxivDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.data = []
        if self.args['mode'] == 'train':
            self.path_name = path
            rar_path = f'{args["root_dir"]}/data/{args["dataset"]}/train.rar'
            if os.path.exists(rar_path):
                self.reader = torch.load(rar_path)
                print(f'[!] load RandomAccessReader Object over')
            else:
                # init the random access reader
                self.reader = RandomAccessReader(self.path_name)
                # this command may take a long time (just wait)
                self.reader.init()
                torch.save(self.reader, rar_path)
                print(f'[!] save the random access reader file into {rar_path}')
            self.reader.init_file_handler()
            self.size = self.reader.size
            print(f'[!] dataset size: {self.size}')
        else:
            data = read_json_data_arxiv(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context).strip()
                self.data.append({
                    'label': [1] + [0] * 9,
                    'context': context,
                    'responses': [response] + candidates,
                })
            self.size = len(self.data)

    def _length_limit(self, ids):
        # also return the speaker embeddings
        if len(ids) > self.args['max_len']:
            ids = [ids[0]] + ids[-(self.args['max_len']-1):]
        return ids
    
    def _length_limit_res(self, ids):
        # cut tail
        if len(ids) > self.args['res_max_len']:
            ids = ids[:self.args['res_max_len']-1] + [self.sep]
        return ids
                
    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            line = self.reader.get_line(i)
            line = json.loads(line.strip())
            context = ' [SEP] '.join(line['q'])
            response = line['r']
            return context, response
        else:
            bundle = self.data[i]
            context = bundle['context']
            responses = bundle['responses']
            return context, responses, bundle['label']

    def save(self):
        pass
        
    def generate_mask(self, ids):
        attn_mask_index = ids.nonzero().tolist()   # [PAD] IS 0
        attn_mask_index_x, attn_mask_index_y = [i[0] for i in attn_mask_index], [i[1] for i in attn_mask_index]
        attn_mask = torch.zeros_like(ids)
        attn_mask[attn_mask_index_x, attn_mask_index_y] = 1
        return attn_mask
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            context = [i[0] for i in batch]
            responses = [i[1] for i in batch]
            return {
                'context': context, 
                'responses': responses, 
            }
        else:
            assert len(batch) == 1
            batch = batch[0]
            context, responses, label = batch[0], batch[1], batch[2]
            label = torch.LongTensor(label)
            if torch.cuda.is_available():
                label = label.cuda()
            return {
                'context': context, 
                'responses': responses, 
                'label': label,
                'text': [responses[0]],    # this item will be used for test_recall script
            }
