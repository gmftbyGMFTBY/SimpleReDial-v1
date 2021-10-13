from .header import *
from dataloader.util_func import *
from .utils import *

'''
Base Agent
'''

class RetrievalBaseAgent:

    def __init__(self):
        # open the test save scores file handler
        self.best_test = None 
        self.checkpointadapeter = CheckpointAdapter()

    def show_parameters(self, args):
        print(f'========== Model Parameters ==========')
        for key, value in args.items():
            if key in ['models', 'deploy', 'datasets', 'no_test_models', 'no_train_models']:
                # too long don't show
                continue
            print(f'{key}: {value}')
        print(f'========== Model Parameters ==========')

    def save_model(self, path):
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
    
    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter):
        raise NotImplementedError

    def set_test_interval(self):
        self.args['test_step'] = [int(self.args['total_step']*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0
        print(f'[!] test interval steps: {self.args["test_step"]}')

    def compare_performance(self, new_test):
        if self.best_test is None:
            self.best_test = new_test
            return True

        r10_1 = self.best_test['R10@1']
        r10_2 = self.best_test['R10@2']
        r10_5 = self.best_test['R10@5']
        avg_mrr = self.best_test['MRR']
        avg_p1 = self.best_test['P@1']
        avg_map = self.best_test['MAP']
        now_test_score = r10_1 + r10_2 + r10_5 + avg_mrr + avg_p1 + avg_map 
        
        r10_1 = new_test['R10@1']
        r10_2 = new_test['R10@2']
        r10_5 = new_test['R10@5']
        avg_mrr = new_test['MRR']
        avg_p1 = new_test['P@1']
        avg_map = new_test['MAP']
        new_test_score = r10_1 + r10_2 + r10_5 + avg_mrr + avg_p1 + avg_map 

        if new_test_score > now_test_score:
            self.best_test = new_test
            return True
        else:
            return False

    def test_now(self, test_iter, recoder):
        # if the model is the no-test-model, save the checkpoint and return
        if self.args['model'] in self.args['no_test_models']:
            if self.args['local_rank'] == 0:
                pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
                # save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{pretrained_model_name}.pt'
                # PAUSE: add the version
                save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{pretrained_model_name}_{self.args["version"]}.pt'
                self.save_model(save_path)
            return

        index = self.test_step_counter
        test_rest = self.test_model(test_iter)

        print(test_rest)

        r10_1 = test_rest['R10@1']
        r10_2 = test_rest['R10@2']
        r10_5 = test_rest['R10@5']
        avg_mrr = test_rest['MRR']
        avg_p1 = test_rest['P@1']
        avg_map = test_rest['MAP']

        if recoder:
            recoder.add_scalar(f'train-test/R10@1', r10_1, index)
            recoder.add_scalar(f'train-test/R10@2', r10_2, index)
            recoder.add_scalar(f'train-test/R10@5', r10_5, index)
            recoder.add_scalar(f'train-test/MRR', avg_mrr, index)
            recoder.add_scalar(f'train-test/P@1', avg_p1, index)
            recoder.add_scalar(f'train-test/MAP', avg_map, index)
        self.test_step_counter += 1
        
        # find the new best model, save
        if self.args['local_rank'] == 0:
            # check the performance
            if self.compare_performance(test_rest):
                pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
                # save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{pretrained_model_name}.pt'
                save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{pretrained_model_name}_{self.args["version"]}.pt'
                self.save_model(save_path)
                print(f'[!] find new best model at test step: {index}')

        self.model.train()    # reset the train mode

        # if have stop_train ans activate
        if 'stop_train' in self.args and self.args['stop_train']:
            for key, value in self.args['stop_train_trigger'].items():
                if test_rest[key] >= value:
                    # ealy stop
                    print(f'[!] training stop at test step: {index}')
                    exit()

    def load_checkpoint(self):
        if 'checkpoint' in self.args:
            if self.args['checkpoint']['is_load']:
                path = self.args['checkpoint']['path']
                path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{path}'
                self.load_model(path)
                print(f'[!] load checkpoint from {path}')
            else:
                print(f'[!] DONOT load checkpoint')
        else:
            print(f'[!] No checkpoint information found')

    def set_optimizer_scheduler_ddp(self):
        if self.args['mode'] in ['train']:
            self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.scaler = GradScaler()
            # self.scheduler = transformers.get_cosine_with_hard_restarts_schedule_with_warmup(
            #     self.optimizer, 
            #     num_warmup_steps=self.args['warmup_step'], 
            #     num_training_steps=self.args['total_step'],
            #     num_cycles=self.args['num_cycles'],
            # )
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_step'], 
                num_training_steps=self.args['total_step'],
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args['local_rank']], 
                output_device=self.args['local_rank'],
                find_unused_parameters=True,
            )
        elif self.args['mode'] in ['inference']:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args['local_rank']], 
                output_device=self.args['local_rank'],
                find_unused_parameters=True,
            )
        else:
            # test doesn't need DDP
            pass

    def load_model(self, path):
        # for test and inference, just load them all
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.checkpointadapeter.init(
            state_dict.keys(),
            self.model.state_dict().keys(),
        )
        new_state_dict = self.checkpointadapeter.convert(state_dict)
        self.model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')

    def convert_to_text(self, ids, lang='zh'):
        '''convert to text and ignore the padding token;
        no [CLS] and no the latest [SEP]'''
        if lang == 'zh':
            sep = '' if lang == 'zh' else ' '
            tokens = [self.vocab.convert_ids_to_tokens(i) for i in ids.cpu().tolist() if i != self.vocab.pad_token_id]
            text = sep.join(tokens)
        else:
            text = self.vocab.decode(ids).replace('[PAD]', '').strip()
        text = text.replace('[SEP]', ' [SEP] ').replace('[CLS]', '').replace('[UNK]', ' [UNK] ')
        text = text.strip(' [SEP] ')
        return text

    @torch.no_grad()
    def rerank(self, contexts, candidates):
        raise NotImplementedError

    def totensor(self, texts, ctx=True, position=False):
        if ctx:
            if type(texts[0]) == list:
                if position is False:
                    ids = []
                    for text in texts:
                        item = self.vocab.batch_encode_plus(text, add_special_tokens=False)['input_ids']
                        context = []
                        for u in item:
                            context.extend(u+[self.eos])
                        context.pop()
                        context = context[-(self.args['max_len']-2):]
                        context = [self.cls] + context + [self.sep]
                        ids.append(torch.LongTensor(context))
                else:
                    ids = []
                    pos_w = []
                    for text in texts:
                        item = self.vocab.batch_encode_plus(text, add_special_tokens=False)['input_ids']
                        context = []
                        pos = []
                        w = self.args['min_w']
                        for u in item:
                            context.extend(u+[self.eos])
                            pos.extend([w]*(len(u)+1))
                            w += self.args['w_delta']
                        context.pop()
                        pos.pop()
                        context = context[-(self.args['max_len']-2):]
                        pos = pos[-(self.args['max_len']-2):]
                        context = [self.cls] + context + [self.sep]
                        pos = [self.args['min_w']] + pos + [w - self.args['w_delta']]
                        ids.append(torch.LongTensor(context))
                        pos_w.append(torch.tensor(pos))
            else:
                items = self.vocab.batch_encode_plus(texts)['input_ids']
                ids = [torch.LongTensor(length_limit(i, self.args['max_len'])) for i in items]
        else:
            items = self.vocab.batch_encode_plus(texts)['input_ids']
            ids = [torch.LongTensor(length_limit_res(i, self.args['res_max_len'], sep=self.sep)) for i in items]
        if position is False:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)
            return ids, mask
        else:
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            pos_w = pad_sequence(pos_w, batch_first=True, padding_value=0.)
            ids, mask, pos_w = to_cuda(ids, mask, pos_w)
            return ids, mask, pos_w

    def totensor_interaction(self, ctx_, responses_):
        '''for Interaction Models'''
        def _encode_one_session(ctx, responses):
            context_length = len(ctx)
            utterances = self.vocab.batch_encode_plus(ctx + responses, add_special_tokens=False)['input_ids']
            context_utterances = utterances[:context_length]
            response_utterances = utterances[context_length:]

            context = []
            for u in context_utterances:
                context.extend(u + [self.eos])
            context.pop()
    
            ids, tids = [], []
            for res in response_utterances:
                ctx = deepcopy(context)
                truncate_pair(ctx, res, self.args['max_len'])
                ids_ = [self.cls] + ctx + [self.sep] + res + [self.sep]
                tids_ = [0] * (len(ctx) + 2) + [1] * (len(res) + 1)
                ids.append(torch.LongTensor(ids_))
                tids.append(torch.LongTensor(tids_))
            return ids, tids

        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        ids, tids = [], []
        for idx in range(0, len(responses_), 128):
            responses = responses_[idx:idx+128]
            ids_, tids_ = _encode_one_session(ctx_, responses)
            ids.extend(ids_)
            tids.extend(tids_)
        ids = pad_sequence(ids_, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids_, batch_first=True, padding_value=self.pad)
        mask = generate_mask(ids)
        ids, tids, mask = to_cuda(ids, tids, mask)
        return ids, tids, mask


class GenerationBaseAgent:

    def __init__(self):
        # open the test save scores file handler
        self.best_test = None 
        self.checkpointadapeter = CheckpointAdapter()

    def show_parameters(self, args):
        print(f'========== Model Parameters ==========')
        for key, value in args.items():
            if key in ['models', 'deploy', 'datasets', 'no_test_models', 'no_train_models']:
                # too long don't show
                continue
            print(f'{key}: {value}')
        print(f'========== Model Parameters ==========')

    def save_model(self, path):
        try:
            state_dict = self.model.module.state_dict()
        except:
            state_dict = self.model.state_dict()
        torch.save(state_dict, path)
        print(f'[!] save model into {path}')
    
    def train_model(self, train_iter, mode='train'):
        raise NotImplementedError

    def test_model(self, test_iter):
        raise NotImplementedError

    def set_test_interval(self):
        self.args['test_step'] = [int(self.args['total_step']*i) for i in np.arange(0, 1+self.args['test_interval'], self.args['test_interval'])]
        self.test_step_counter = 0
        print(f'[!] test interval steps: {self.args["test_step"]}')

    def compare_performance(self, new_test):
        if self.best_test is None:
            self.best_test = new_test
            return True
        ppl_pos = self.best_test['PPL-pos']
        ppl_neg = self.best_test['PPL-neg']
        b_1 = self.best_test['BLEU-1']
        b_2 = self.best_test['BLEU-2']
        b_3 = self.best_test['BLEU-3']
        b_4 = self.best_test['BLEU-4']
        rouge_l = self.best_test['ROUGE-L']
        meteor = self.best_test['METEOR']
        p = self.best_test['BERTScore-P']
        r = self.best_test['BERTScore-R']
        f = self.best_test['BERTScore-F']
        # now_test_score = b_1 + b_2 + b_3 + b_4 + rouge_l + meteor + p + r + f
        now_test_score = ppl_neg - ppl_pos
        
        ppl_pos = new_test['PPL-pos']
        ppl_neg = new_test['PPL-neg']
        b_1 = new_test['BLEU-1']
        b_2 = new_test['BLEU-2']
        b_3 = new_test['BLEU-3']
        b_4 = new_test['BLEU-4']
        rouge_l = new_test['ROUGE-L']
        meteor = new_test['METEOR']
        p = new_test['BERTScore-P']
        r = new_test['BERTScore-R']
        f = new_test['BERTScore-F']
        # new_test_score = b_1 + b_2 + b_3 + b_4 + rouge_l + meteor + p + r + f
        new_test_score = ppl_neg - ppl_pos
        if new_test_score > now_test_score:
            self.best_test = new_test
            return True
        else:
            return False

    def test_now(self, test_iter, recoder):
        # if the model is the no-test-model, save the checkpoint and return
        if self.args['model'] in self.args['no_test_models']:
            if self.args['local_rank'] == 0:
                pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
                save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{pretrained_model_name}.pt'
                self.save_model(save_path)
            return

        index = self.test_step_counter
        test_rest = self.test_model(test_iter)

        print(test_rest)
        ppl_pos = test_rest['PPL-pos']
        ppl_neg = test_rest['PPL-neg']
        b_1 = test_rest['BLEU-1']
        b_2 = test_rest['BLEU-2']
        b_3 = test_rest['BLEU-3']
        b_4 = test_rest['BLEU-4']
        rouge_l = test_rest['ROUGE-L']
        meteor = test_rest['METEOR']
        p = test_rest['BERTScore-P']
        r = test_rest['BERTScore-R']
        f = test_rest['BERTScore-F']

        if recoder:
            recoder.add_scalar(f'train-test/PPL-pos', ppl_pos, index)
            recoder.add_scalar(f'train-test/PPL-neg', ppl_neg, index)
            recoder.add_scalar(f'train-test/BLEU-1', b_1, index)
            recoder.add_scalar(f'train-test/BLEU-2', b_2, index)
            recoder.add_scalar(f'train-test/BLEU-3', b_3, index)
            recoder.add_scalar(f'train-test/BLEU-4', b_4, index)
            recoder.add_scalar(f'train-test/ROUGE-L', rouge_l, index)
            recoder.add_scalar(f'train-test/METEOR', meteor, index)
            recoder.add_scalar(f'train-test/BERTScore-P', p, index)
            recoder.add_scalar(f'train-test/BERTScore-R', r, index)
            recoder.add_scalar(f'train-test/BERTScore-F', f, index)
        self.test_step_counter += 1
        
        # find the new best model, save
        if self.args['local_rank'] == 0:
            # check the performance
            if self.compare_performance(test_rest):
                pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
                save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{pretrained_model_name}.pt'
                self.save_model(save_path)
                print(f'[!] find new best model at test step: {index}')

        self.model.train()    # reset the train mode

        # if have stop_train as activate
        if 'stop_train' in self.args and self.args['stop_train']:
            for key, value in self.args['stop_train_trigger'].items():
                if test_rest[key] >= value:
                    # ealy stop
                    print(f'[!] training stop at test step: {index}')
                    exit()

    def load_checkpoint(self):
        if 'checkpoint' in self.args:
            if self.args['checkpoint']['is_load']:
                path = self.args['checkpoint']['path']
                path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{path}'
                self.load_model(path)
                print(f'[!] load checkpoint from {path}')
            else:
                print(f'[!] DONOT load checkpoint')
        else:
            print(f'[!] No checkpoint information found')

    def set_optimizer_scheduler_ddp(self):
        if self.args['mode'] in ['train']:
            self.optimizer = transformers.AdamW(
                self.model.parameters(), 
                lr=self.args['lr'],
            )
            self.scaler = GradScaler()
            self.scheduler = transformers.get_linear_schedule_with_warmup(
                self.optimizer, 
                num_warmup_steps=self.args['warmup_step'], 
                num_training_steps=self.args['total_step'],
            )
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args['local_rank']], 
                output_device=self.args['local_rank'],
                find_unused_parameters=True,
            )
        elif self.args['mode'] in ['inference']:
            self.model = nn.parallel.DistributedDataParallel(
                self.model, 
                device_ids=[self.args['local_rank']], 
                output_device=self.args['local_rank'],
                find_unused_parameters=True,
            )
        else:
            # test doesn't need DDP
            pass

    def load_model(self, path):
        # for test and inference, just load them all
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.checkpointadapeter.init(
            state_dict.keys(),
            self.model.state_dict().keys(),
        )
        new_state_dict = self.checkpointadapeter.convert(state_dict)
        self.model.load_state_dict(new_state_dict)
        print(f'[!] load model from {path}')
