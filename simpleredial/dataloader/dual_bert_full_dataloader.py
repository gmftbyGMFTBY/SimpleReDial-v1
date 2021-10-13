from header import *
from .utils import *
from .randomaccess import *
from .util_func import *


class BERTDualFullWithNegDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.data = []
        if self.args['mode'] == 'train':
            self.path_name = path
            responses = []
            with open(path) as f:
                pbar = tqdm(f.readlines())
                for line in pbar:
                    line = [i.strip() for i in json.loads(line.strip())['nr'] if i.strip()]
                    responses.extend(line)
                    if len(responses) > 1000000:
                        break
                    pbar.set_description(f'[!] already collect {len(responses)} utterances for candidates')
            self.responses = list(set(responses))
            print(f'[!] load {len(self.responses)} utterances')
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
            data, responses = read_json_data(path, lang=self.args['lang'])
            for context, response, candidates in tqdm(data):
                context = ' [SEP] '.join(context).strip()
                if len(candidates) < 9:
                    candidates += random.sample(responses, 9-len(candidates))
                else:
                    candidates = candidates[:9]
                self.data.append({
                    'label': [1] + [0] * 9,
                    'context': context,
                    'responses': [response] + candidates,
                })
            self.size = len(self.data)

    def __len__(self):
        return self.size

    def __getitem__(self, i):
        if self.args['mode'] == 'train':
            line = self.reader.get_line(i)
            line = json.loads(line.strip())
            context = ' [SEP] '.join(line['q'])
            response = line['r']
            candidates = [i for i in line['nr'] if i.strip()]
            if len(candidates) < self.args['gray_cand_num']:
                candidates += random.sample(self.responses, self.args['gray_cand_num']-len(candidates))
            else:
                candidates = random.sample(candidates, self.args['gray_cand_num'])
            responses = [response] + candidates
            return context, responses
        else:
            bundle = self.data[i]
            context = bundle['context']
            responses = bundle['responses']
            return context, responses, bundle['label']

    def save(self):
        pass
        
    def collate(self, batch):
        if self.args['mode'] == 'train':
            context = [i[0] for i in batch]
            responses = []
            for i in batch:
                responses.extend(i[1])
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
