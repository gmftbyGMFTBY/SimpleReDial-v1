from model.utils import *


class RAGenerationAgent(GenerationBaseAgent):

    '''Retrieval augmented Generation Models'''
    
    def __init__(self, vocab, model, args):
        super(RAGenerationAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        
        if args['mode'] == 'train':
            self.set_test_interval()
            self.load_checkpoint()

        # open the test save scores file handler
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}.txt'
        self.log_save_file = open(path, 'w')

        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)
        # metrics
        self.nlgeval = NLGEval(no_glove=True, no_skipthoughts=True)
        self.bertscorer = BERTScorer(lang=self.args['lang'], rescale_with_baseline=True)
    
    def train_model(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, total_token_acc, batch_num = 0, 0, 0
        total_ppl = 0
        pbar = tqdm(train_iter)
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                loss, token_acc = self.model(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            ppl = math.exp(loss.item())
            total_ppl += ppl
            total_loss += loss.item()
            total_token_acc += token_acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/PPL', total_ppl/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunPPL', ppl, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', total_token_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', token_acc, idx)
             
            pbar.set_description(f'[!] loss: {round(total_loss/batch_num, 4)}; ppl: {round(total_ppl/batch_num, 2)}; token acc: {round(total_token_acc/batch_num*100, 2)}')
        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/PPL', total_ppl/batch_num, idx_)
        recoder.add_scalar(f'train-whole/TokenAcc', total_token_acc/batch_num, idx_)
    
    @torch.no_grad()
    def test_model(self, test_iter, print_output=True):
        self.model.eval()
        pbar = tqdm(test_iter)
        ppl_pos, ppl_neg, p, r, f, bleu_1, bleu_2, bleu_3, bleu_4, rouge_l, meteor = [], [], [], [], [], [], [], [], [], [], []
        results = []
        for idx, batch in enumerate(pbar):            
            if self.args['mode'] == 'train':
                logits = self.model.module.predict(batch)     # [B, S, V]
                # calculate ppl
                ppl_pos_ = self.model.module.calculate_ppl(batch['pos_ids'], batch['pos_ids_mask'])
                ppl_neg_ = self.model.module.calculate_ppl(batch['neg_ids'], batch['neg_ids_mask'])
            else:
                logits = self.model.predict(batch)     # [B, S, V]
                # calculate ppl
                ppl_pos_ = self.model.calculate_ppl(batch['pos_ids'], batch['pos_ids_mask'])
                ppl_neg_ = self.model.calculate_ppl(batch['neg_ids'], batch['neg_ids_mask'])
            ppl_pos.append(ppl_pos_)
            ppl_neg.append(ppl_neg_)

            gen_texts = []
            for logit in logits:
                tokens = [i for i in self.vocab.convert_ids_to_tokens(logit) if i not in ['[PAD]', '[CLS]', '[SEP]']]
                gen_texts.append(''.join(tokens))
            if print_output:
                for prefix_t, pos_t, neg_t, gen_t in zip(batch['text'], batch['pos_text'], batch['neg_text'], gen_texts):
                    self.log_save_file.write(f'[Prefix     ] {prefix_t}\n')
                    self.log_save_file.write(f'[Positive   ] {pos_t}\n')
                    self.log_save_file.write(f'[Negative   ] {neg_t}\n')
                    self.log_save_file.write(f'[Generation ] {gen_t}\n\n')
            for gt_t, gen_t in zip(batch['text'], gen_texts):
                results.append((gt_t, gen_t))
        # calculate the evalution results
        inner_bsz = 64
        for idx in tqdm(range(0, len(results), inner_bsz)):
            candidates = [i[1] for i in results[idx:idx+inner_bsz]]
            references = [i[0] for i in results[idx:idx+inner_bsz]]
            P, R, F = self.bertscorer.score(candidates, references)
            p.extend(P.tolist())
            r.extend(R.tolist())
            f.extend(F.tolist())

            for candidate, reference in zip(candidates, references):
                candidate, reference = ' '.join(list(candidate)), ' '.join(list(reference))
                rest = self.nlgeval.compute_individual_metrics(ref=[reference], hyp=candidate)
                bleu_1.append(rest['Bleu_1'])
                bleu_2.append(rest['Bleu_2'])
                bleu_3.append(rest['Bleu_3'])
                bleu_4.append(rest['Bleu_4'])
                rouge_l.append(rest['ROUGE_L'])
                meteor.append(rest['METEOR'])
        ppl_pos = np.mean(ppl_pos)
        ppl_neg = np.mean(ppl_neg)
        p_ = np.mean(p)
        r_ = np.mean(r)
        f_ = np.mean(f)
        b_1 = np.mean(bleu_1)
        b_2 = np.mean(bleu_2)
        b_3 = np.mean(bleu_3)
        b_4 = np.mean(bleu_4)
        r_l = np.mean(rouge_l)
        meteor_ = np.mean(meteor)
        return {
            'PPL-pos': ppl_pos,
            'PPL-neg': ppl_neg,
            'BLEU-1': b_1,
            'BLEU-2': b_2,
            'BLEU-3': b_3,
            'BLEU-4': b_4,
            'ROUGE-L': r_l,
            'METEOR': meteor_,
            'BERTScore-P': p_,
            'BERTScore-R': r_,
            'BERTScore-F': f_,
        }

    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['model'] in ['gpt2']:
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.model.load_state_dict(new_state_dict)
        else:
            if self.args['mode'] == 'train':
                # the context encoder model has been loaded (GPT-2)
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.can_encoder.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.can_encoder.load_state_dict(new_state_dict)
            else:
                # test and inference mode
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.load_state_dict(new_state_dict)
            print(f'[!] load model from {path}')
