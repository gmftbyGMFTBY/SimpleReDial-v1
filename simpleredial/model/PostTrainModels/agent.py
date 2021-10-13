from model.utils import *

class PostTrainAgent(RetrievalBaseAgent):

    def __init__(self, vocab, model, args):
        super(PostTrainAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model

        # special token [EOS]
        special_tokens_dict = {'eos_token': '[EOS]'}
        self.vocab.add_special_tokens(special_tokens_dict)

        if args['mode'] == 'train':
            self.set_test_interval()
            self.load_checkpoint()
        else:
            # open the test save scores file handler
            pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}.txt'
            self.log_save_file = open(path, 'w')
        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        if args['model'] in ['simcse']:
            self.train_model = self.train_model_simcse
        self.show_parameters(self.args)

        # best metric (acc)
        self.best_acc = -1
        
    def train_model(self, train_iter, test_iter, recoder=None, idx_=0, whole_batch_num=0):
        self.model.train()

        total_loss, batch_num = 0, 0
        total_mlm_loss, total_cls_loss = 0, 0
        total_cls_acc, total_mlm_acc = 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                mlm_loss, cls_loss, token_acc, cls_acc = self.model(batch)
                loss = mlm_loss + cls_loss
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_loss += loss.item()
            total_cls_loss += cls_loss.item()
            total_mlm_loss += mlm_loss.item()
            total_cls_acc += cls_acc
            total_mlm_acc += token_acc
            batch_num += 1
           
            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/CLSLoss', total_cls_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunCLSLoss', cls_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/MLMLoss', total_mlm_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunMLMLoss', mlm_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/TokenAcc', total_mlm_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunTokenAcc', token_acc, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_cls_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', cls_acc, idx)
            pbar.set_description(f'[!] loss: {round(total_cls_loss/batch_num, 2)}|{round(total_mlm_loss/batch_num, 2)}; acc: {round(100*total_cls_acc/batch_num, 2)}|{round(100*total_mlm_acc/batch_num, 2)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/CLSLoss', total_cls_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/MLMLoss', total_mlm_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/TokenAcc', total_mlm_acc/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_cls_acc/batch_num, idx_)

        # current acc
        current_acc = total_mlm_acc/batch_num + total_cls_acc/batch_num
        if current_acc > self.best_acc:
            self.best_acc = current_acc
            # save the model
            if self.args['local_rank'] == 0:
                pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
                save_path = f'{self.args["root_dir"]}/ckpt/{self.args["dataset"]}/{self.args["model"]}/best_{pretrained_model_name}.pt'
                self.save_model(save_path)
        return batch_num
    
    def train_model_simcse(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                loss, acc = self.model(batch)
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)

    @torch.no_grad()
    def test_model(self, test_iter, print_output=False, rerank_agent=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]

        if self.args['model'] in ['simcse']:
            return

        for idx, batch in enumerate(pbar):                
            label = batch['label']
            if 'context' in batch:
                cid, cid_mask = self.totensor([batch['context']], ctx=True)
                rid, rid_mask = self.totensor(batch['responses'], ctx=False)
                batch['ids'], batch['ids_mask'] = cid, cid_mask
                batch['rids'], batch['rids_mask'] = rid, rid_mask
            elif 'ids' in batch:
                cid = batch['ids'].unsqueeze(0)
                cid_mask = torch.ones_like(cid)
                batch['ids'] = cid
                batch['ids_mask'] = cid_mask

            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(batch).cpu().tolist()    # [B]
            else:
                scores = self.model.predict(batch).cpu().tolist()    # [B]

            # rerank by the compare model (bert-ft-compare)
            if rerank_agent:
                if 'context' in batch:
                    context = batch['context']
                    responses = batch['responses']
                elif 'ids' in batch:
                    context = self.convert_to_text(batch['ids'].squeeze(0), lang=self.args['lang'])
                    responses = [self.convert_to_text(res, lang=self.args['lang']) for res in batch['rids']]
                packup = {
                    'context': context,
                    'responses': responses,
                    'scores': scores,
                }
                # only the scores has been update
                # scores = rerank_agent.compare_reorder(packup)
                scores = rerank_agent.compare_reorder_fast(packup)

            # print output
            if print_output:
                if 'responses' in batch:
                    self.log_save_file.write(f'[CTX] {batch["context"]}\n')
                    for rtext, score in zip(responses, scores):
                        score = round(score, 4)
                        self.log_save_file.write(f'[Score {score}] {rtext}\n')
                else:
                    ctext = self.convert_to_text(batch['ids'].squeeze(0))
                    self.log_save_file.write(f'[CTX] {ctext}\n')
                    for rid, score in zip(batch['rids'], scores):
                        rtext = self.convert_to_text(rid)
                        score = round(score, 4)
                        self.log_save_file.write(f'[Score {score}] {rtext}\n')
                self.log_save_file.write('\n')

            rank_by_pred, pos_index, stack_scores = \
            calculate_candidates_ranking(
                np.array(scores), 
                np.array(label.cpu().tolist()),
                10)
            num_correct = logits_recall_at_k(pos_index, k_list)
            if self.args['dataset'] in ["douban"]:
                total_prec_at_one += precision_at_one(rank_by_pred)
                total_map += mean_average_precision(pos_index)
                for pred in rank_by_pred:
                    if sum(pred) == 0:
                        total_examples -= 1
            total_mrr += logits_mrr(pos_index)
            total_correct = np.add(total_correct, num_correct)
            total_examples += 1
        avg_mrr = float(total_mrr / total_examples)
        avg_prec_at_one = float(total_prec_at_one / total_examples)
        avg_map = float(total_map / total_examples)
        return {
            f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
            f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
            f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
            'MRR': round(avg_mrr, 4),
            'P@1': round(avg_prec_at_one, 4),
            'MAP': round(avg_map, 4),
        }
    
    @torch.no_grad()
    def inference_simcse_unlikelyhood_ctx(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts, indexes, responses = [], [], [], []
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['mask']
            res = self.model.module.get_embedding(ids, ids_mask).cpu()
            embds.append(res)
            texts.extend(batch['context'])
            responses.extend(batch['response'])
            indexes.extend(batch['index'])
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            res = responses[i:i+size]
            index = indexes[i:i+size]
            torch.save(
                (embd, text, res, index), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_simcse_ctx_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference_simcse_ctx(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts, indexes = [], [], []
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['mask']
            text = batch['text']
            res = self.model.module.get_embedding(ids, ids_mask).cpu()
            embds.append(res)
            texts.extend(text)
            indexes.extend(batch['index'])
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            index = indexes[i:i+size]
            torch.save(
                (embd, text, index), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_simcse_ctx_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts, contexts = [], [], []
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['mask']
            text = batch['text']
            res = self.model.module.get_embedding(ids, ids_mask).cpu()
            embds.append(res)
            texts.extend(text)
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )

    def load_model(self, path):
        if self.args['mode'] == 'train':
            if self.args['model'] in ['simcse']:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.encoder.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.encoder.load_state_dict(new_state_dict)
                print(f'[!] simcse loads pre-trained model from {path}')
            elif self.args['model'] in ['bert-fp-mono']:
                if self.args['dataset'] in ['restoration-200k']:
                    state_dict = torch.load(path, map_location=torch.device('cpu'))
                    self.checkpointadapeter.init(
                        state_dict.keys(),
                        self.model.state_dict().keys(),
                    )
                    new_state_dict = self.checkpointadapeter.convert(state_dict)
                    self.model.load_state_dict(new_state_dict)
                    print(f'[!] bert-fp-mono loads pre-trained model from {path}')
                else:
                    state_dict = torch.load(path, map_location=torch.device('cpu'))
                    self.checkpointadapeter.init(
                        state_dict.keys(),
                        self.model.model.bert.state_dict().keys(),
                    )
                    new_state_dict = self.checkpointadapeter.convert(state_dict)
                    self.model.model.bert.load_state_dict(new_state_dict)

            elif self.args['model'] in ['dual-bert-unsup']:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.ctx_encoder.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.ctx_encoder.load_state_dict(new_state_dict)
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.can_encoder.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.can_encoder.load_state_dict(new_state_dict)
            elif self.args['model'] in ['dual-bert-pt']:
                state_dict = torch.load(path, map_location=torch.device('cpu'))
                new_ctx_state_dict = OrderedDict()
                new_can_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'cls.seq_relationship' in k:
                        pass
                    elif 'bert.pooler' in k:
                        pass
                    else:
                        k = k.lstrip('model.')
                        new_ctx_state_dict[k] = v
                        new_can_state_dict[k] = v
                self.model.ctx_encoder.load_state_dict(new_ctx_state_dict)
                self.model.can_encoder.load_state_dict(new_can_state_dict)
        else:
            # test or inference
            state_dict = torch.load(path, map_location=torch.device('cpu'))
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.load_state_dict(new_state_dict)
            print(f'[!] Inference mode: simcse loads pre-trained model from {path}')
