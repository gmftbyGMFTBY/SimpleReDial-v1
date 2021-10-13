from model.utils import *
from dataloader.util_func import *

class RepresentationAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(RepresentationAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.vocab.add_tokens(['[EOS]'])

        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')

        if args['mode'] == 'train':
            # hash-bert parameter setting
            if self.args['model'] in ['hash-bert']:
                self.q_alpha = self.args['q_alpha']
                self.q_alpha_step = (self.args['q_alpha_max'] - self.args['q_alpha']) / int(self.args['total_step'] / torch.distributed.get_world_size())
                self.train_model = self.train_model_hash
            elif self.args['model'] in ['dual-bert-ssl']:
                self.train_model = self.train_model_ssl
                # set hyperparameters
                self.model.ssl_interval_step = int(self.args['total_step'] * self.args['ssl_interval'])
            elif self.args['model'] in ['dual-bert-pt']:
                self.train_model = self.train_model_pt
            elif self.args['model'] in ['dual-bert-adv']:
                self.train_model = self.train_model_adv

            self.set_test_interval()
            self.load_checkpoint()
        else:
            # open the test save scores file handler
            pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
            path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}_{args["version"]}.txt'
            self.log_save_file = open(path, 'w')
            if args['model'] in ['dual-bert-fusion']:
                self.inference = self.inference2
                print(f'[!] switch the inference function')
            elif args['model'] in ['dual-bert-one2many']:
                self.inference = self.inference_one2many
                print(f'[!] switch the inference function')
        if torch.cuda.is_available():
            self.model.cuda()
        if args['mode'] in ['train', 'inference']:
            self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)
        # Metrics object
        self.metrics = Metrics()

        if self.args['fgm']:
            self.fgm = FGM(self.model)

    def train_model_hash(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, batch_num = 0, 0
        total_h_loss, total_q_loss, total_kl_loss = 0, 0, 0
        total_acc = 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        for idx, batch in enumerate(pbar):
            self.optimizer.zero_grad()
            with autocast():
                kl_loss, hash_loss, quantization_loss, acc = self.model(batch)
                quantization_loss *= self.q_alpha
                loss = kl_loss + hash_loss + quantization_loss
                self.q_alpha += self.q_alpha_step
            self.scaler.scale(loss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            
            self.scheduler.step()

            total_loss += loss.item()
            total_kl_loss += kl_loss.item()
            total_q_loss += quantization_loss.item()
            total_h_loss += hash_loss.item()
            total_acc += acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
            
            recoder.add_scalar(f'train-epoch-{idx_}/q_alpha', self.q_alpha, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/KLLoss', total_kl_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunKLLoss', kl_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/HashLoss', total_h_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunHashLoss', hash_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/QuantizationLoss', total_q_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunQuantizationLoss', quantization_loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] kl_loss: {round(kl_loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')

        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/KLLoss', total_kl_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/QLoss', total_q_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/HLoss', total_h_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
    def train_model_ssl(self, train_iter, test_iter, recoder=None, idx_=0):
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

            # update may copy the parameters from original model to shadow model
            self.model.module.update()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)
            
            recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
            recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
            recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
             
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc: {round(acc, 4)}|{round(total_acc/batch_num, 4)}')

        recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
        recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
    def train_model_pt(self, train_iter, test_iter, recoder=None, idx_=0):
        '''for dual-bert-pt model'''
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

            if batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

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
             
            pbar.set_description(f'[!] loss: {round(cls_loss.item(), 4)}|{round(total_cls_loss/batch_num, 4)}; acc: {round(cls_acc, 4)}|{round(total_cls_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/CLSLoss', total_cls_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/MLMLoss', total_mlm_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/TokenAcc', total_mlm_acc/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_cls_acc/batch_num, idx_)
        return round(total_loss / batch_num, 4)
    
    def train_model(self, train_iter, test_iter, recoder=None, idx_=0, hard=False, whole_batch_num=0):
        self.model.train()
        total_loss, total_acc = 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        batch_num = 0
        for idx, batch in enumerate(pbar):

            # compatible with the curriculumn learning
            batch['mode'] = 'hard' if hard is True else 'easy'

            self.optimizer.zero_grad()

            if self.args['model'] in ['dual-bert-gray-writer']:
                cid, cid_mask = self.totensor(batch['context'], ctx=True)
                rid, rid_mask = self.totensor(batch['responses'], ctx=False)
                batch['cid'], batch['cid_mask'] = cid, cid_mask
                batch['rid'], batch['rid_mask'] = rid, rid_mask

            if self.args['fgm']:
                with autocast():
                    loss, acc = self.model(batch)
                self.scaler.scale(loss).backward()
                self.fgm.attack()
                with autocast():
                    loss_adv, _ = self.model(batch)
                self.scaler.scale(loss_adv).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(
                    self.model.parameters(), 
                    self.args['grad_clip']
                )
                self.fgm.restore()
                self.scaler.step(self.optimizer)
                self.scaler.update()
            else:
                with autocast():
                    loss, acc = self.model(batch)
                self.scaler.scale(loss).backward()
                self.scaler.unscale_(self.optimizer)
                clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
                self.scaler.step(self.optimizer)
                self.scaler.update()

            # comment for the constant learning ratio
            self.scheduler.step()

            total_loss += loss.item()
            total_acc += acc
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
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
        return batch_num
   
    @torch.no_grad()
    def test_model(self, test_iter, print_output=False, rerank_agent=None, core_time=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        core_time_rest = 0
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
                if core_time:
                    bt = time.time()
                scores = self.model.predict(batch).cpu().tolist()    # [B]
                if core_time:
                    et = time.time()
                    core_time_rest += et - bt

            # rerank by the compare model (bert-ft-compare)
            if rerank_agent:
                if 'context' in batch:
                    context = batch['context']
                    responses = batch['responses']
                elif 'ids' in batch:
                    context = self.convert_to_text(batch['ids'].squeeze(0), lang=self.args['lang'])
                    responses = [self.convert_to_text(res, lang=self.args['lang']) for res in batch['rids']]
                    context = [i.strip() for i in context.split('[SEP]')]
                packup = {
                    'context': context,
                    'responses': responses,
                    'scores': scores,
                }
                # only the scores has been update
                scores = rerank_agent.compare_reorder(packup)

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
            if self.args['dataset'] in ["douban", "restoration-200k"]:
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
        if core_time:
            return {
                f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
                f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
                f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
                'MRR': round(100*avg_mrr, 2),
                'P@1': round(100*avg_prec_at_one, 2),
                'MAP': round(100*avg_map, 2),
                'core_time': core_time_rest,
            }
        else:
            return {
                f'R10@{k_list[0]}': round(((total_correct[0]/total_examples)*100), 2),        
                f'R10@{k_list[1]}': round(((total_correct[1]/total_examples)*100), 2),        
                f'R10@{k_list[2]}': round(((total_correct[2]/total_examples)*100), 2),        
                'MRR': round(100*avg_mrr, 2),
                'P@1': round(100*avg_prec_at_one, 2),
                'MAP': round(100*avg_map, 2),
            }
    
    @torch.no_grad()
    def inference_writer(self, inf_iter, size=1000000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        source = {}
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = [(t, ti) for t, ti in zip(batch['text'], batch['title'])]
            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text)
            for t, u in zip(batch['title'], batch['url']):
                if t not in source:
                    source[t] = u
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )

        # save sub-source
        torch.save(source, f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_subsource_{self.args["model"]}_{self.args["local_rank"]}.pt')
    
    @torch.no_grad()
    def inference_one2many(self, inf_iter, size=500000):
        '''1 million cut'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = batch['text']
            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text * (self.args['gray_cand_num']+1))
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference_full_ctx_res(self, inf_iter, size=500000):
        '''1 million cut'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        res_embds, ctx_embds, ctexts, rtexts = [], [], [], []
        counter = 0
        for batch in pbar:
            ids = batch['ids']
            rids = batch['rids']
            ids_mask = batch['ids_mask']
            rids_mask = batch['rids_mask']
            ctext = batch['ctext']
            rtext = batch['rtext']
            res = self.model.module.get_cand(rids, rids_mask).cpu()
            ctx = self.model.module.get_ctx(ids, ids_mask).cpu()
            res_embds.append(res)
            ctx_embds.append(ctx)
            ctexts.extend(ctext)
            rtexts.extend(rtext)

            if len(ctexts) > size:
                # save the memory
                res_embds = torch.cat(res_embds, dim=0).numpy()
                ctx_embds = torch.cat(ctx_embds, dim=0).numpy()
                torch.save(
                    (res_embds, ctx_embds, ctexts, rtexts), 
                    f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_full_ctx_res_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
                )
                res_embds, ctx_embds, ctexts, rtexts = [], [], [], []
                counter += 1
        if len(ctexts) > 0:
            res_embds = torch.cat(res_embds, dim=0).numpy()
            ctx_embds = torch.cat(ctx_embds, dim=0).numpy()
            torch.save(
                (res_embds, ctx_embds, ctexts, rtexts), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_full_ctx_res_{self.args["model"]}_{self.args["local_rank"]}_{counter}.pt'
            )
    
    @torch.no_grad()
    def inference_with_source(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts, sources = [], [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = batch['text']
            source = batch['ctext']
            res = self.model.module.get_cand(rid, rid_mask).cpu()
            embds.append(res)
            texts.extend(text)
            sources.extend(source)
        embds = torch.cat(embds, dim=0).numpy()

        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = texts[i:i+size]
            source = sources[i:i+size]
            torch.save(
                (embd, text, source), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_{self.args["model"]}_with_source_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference_data_filter(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        ctext, rtext, s = [], [], []
        for batch in pbar:
            ids = batch['ids']
            rids = batch['rids']
            ids_mask = batch['ids_mask']
            rids_mask = batch['rids_mask']
            cid_rep = self.model.module.get_ctx(ids, ids_mask)
            rid_rep = self.model.module.get_cand(rids, rids_mask)
            scores = (cid_rep * rid_rep).sum(dim=-1).tolist()    # [B]
            ctext.extend(batch['ctext'])
            rtext.extend(batch['rtext'])
            s.extend(scores)
        torch.save(
            (ctext, rtext, s), 
            f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_full_filter_{self.args["model"]}_{self.args["local_rank"]}.pt'
        )

    @torch.no_grad()
    def inference(self, inf_iter, size=500000):
        '''1 million cut'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            text = batch['text']
            res = self.model.module.get_cand(rid, rid_mask).cpu()
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
    
    @torch.no_grad()
    def inference_context_for_response(self, inf_iter, size=1000000):
        '''inference the context for searching the hard negative data'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, responses = [], []
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['mask']
            response = batch['text']
            embd = self.model.module.get_ctx(ids, ids_mask).cpu()
            embds.append(embd)
            responses.extend(response)
        embds = torch.cat(embds, dim=0).numpy()
        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            text = responses[i:i+size]
            torch.save(
                (embd, text), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_context_for_response_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    
    @torch.no_grad()
    def inference_context(self, inf_iter, size=500000):
        '''inference the context for searching the hard negative data'''
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, contexts, responses = [], [], []
        for batch in pbar:
            ids = batch['ids']
            ids_mask = batch['mask']
            context = batch['context']
            response = batch['response']
            embd = self.model.module.get_ctx(ids, ids_mask).cpu()
            embds.append(embd)
            contexts.extend(context)
            responses.extend(response)
        embds = torch.cat(embds, dim=0).numpy()
        for idx, i in enumerate(range(0, len(embds), size)):
            embd = embds[i:i+size]
            context = contexts[i:i+size]
            response = responses[i:i+size]
            torch.save(
                (embd, context, response), 
                f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_context_{self.args["model"]}_{self.args["local_rank"]}_{idx}.pt'
            )
    
    @torch.no_grad()
    def inference2(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        embds, texts = [], []
        for batch in pbar:
            rid = batch['ids']
            rid_mask = batch['mask']
            cid = batch['cid']
            cid_mask = batch['cid_mask']
            text = batch['text']
            res = self.model.module.get_cand(cid, cid_mask, rid, rid_mask).cpu()
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

    @torch.no_grad()
    def encode_queries(self, texts):
        self.model.eval()
        if self.args['model'] in ['dual-bert-pos', 'dual-bert-hn-pos']:
            ids, ids_mask, pos_w = self.totensor(texts, ctx=True, position=True)
            vectors = self.model.get_ctx(ids, ids_mask, pos_w)    # [B, E]
        else:
            ids, ids_mask = self.totensor(texts, ctx=True)
            # vectors = self.model.get_ctx(ids, ids_mask)    # [B, E]
            vectors = self.model.module.get_ctx(ids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()

    @torch.no_grad()
    def encode_candidates(self, texts):
        self.model.eval()
        ids, ids_mask = self.totensor(texts, ctx=False)
        vectors = self.model.get_cand(ids, ids_mask)    # [B, E]
        return vectors.cpu().numpy()

    @torch.no_grad()
    def rerank(self, batches, inner_bsz=2048):
        self.model.eval()
        scores = []
        for batch in batches:
            subscores = []
            # pbar = tqdm(range(0, len(batch['candidates']), inner_bsz))
            cid, cid_mask = self.totensor([batch['context']], ctx=True)
            # for idx in pbar:
            for idx in range(0, len(batch['candidates']), inner_bsz):
                candidates = batch['candidates'][idx:idx+inner_bsz]
                rid, rid_mask = self.totensor(candidates, ctx=False)
                batch['ids'] = cid
                batch['ids_mask'] = cid_mask
                batch['rids'] = rid
                batch['rids_mask'] = rid_mask
                subscores.extend(self.model.predict(batch).tolist())
            scores.append(subscores)
        return scores

    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['mode'] == 'train':
            if 'simsce' in path:
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    self.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.load_state_dict(new_state_dict)
                print(f'[!] load the simcse pre-trained model')
            elif self.args['model'] in ['dual-bert-one2many']:
                new_ctx_state_dict = OrderedDict()
                new_res_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'ctx_encoder' in k:
                        new_k = k.replace('ctx_encoder.', '')
                        new_ctx_state_dict[new_k] = v
                    elif 'can_encoder' in k:
                        new_k = k.replace('can_encoder.', '')
                        new_res_state_dict[new_k] = v
                    else:
                        raise Exception()
                self.model.ctx_encoder.load_state_dict(new_ctx_state_dict)
                self.model.can_encoders[0].load_state_dict(new_res_state_dict)
                self.model.can_encoders[1].load_state_dict(new_res_state_dict)
            elif self.args['model'] in ['dual-bert-hn', 'dual-bert-hn-ctx', 'dual-bert-cl']:
                new_ctx_state_dict = OrderedDict()
                new_res_state_dict = OrderedDict()
                for k, v in state_dict.items():
                    if 'ctx_encoder' in k:
                        new_k = k.replace('ctx_encoder.', '')
                        new_ctx_state_dict[new_k] = v
                    elif 'can_encoder' in k:
                        new_k = k.replace('can_encoder.', '')
                        new_res_state_dict[new_k] = v
                    else:
                        raise Exception()
                self.model.ctx_encoder.load_state_dict(new_ctx_state_dict)
                self.model.can_encoder.load_state_dict(new_res_state_dict)
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
                # context encoder checkpoint
                self.checkpointadapeter.init(
                    state_dict.keys(),
                    # NOTE
                    self.model.ctx_encoder.model.state_dict().keys(),
                )
                new_state_dict = self.checkpointadapeter.convert(state_dict)
                self.model.ctx_encoder.model.load_state_dict(new_state_dict, strict=False)

                if self.args['model'] in ['dual-bert-speaker']:
                    self.model.ctx_encoder_1.model.load_state_dict(new_state_dict, strict=False)

               
                # response encoders checkpoint
                if self.args['model'] in ['dual-bert-multi', 'dual-bert-one2many-original']:
                    for i in range(self.args['gray_cand_num']):
                        self.checkpointadapeter.init(
                            state_dict.keys(),
                            self.model.can_encoders[i].state_dict().keys(),
                        )
                        new_state_dict = self.checkpointadapeter.convert(state_dict)
                        self.model.can_encoders[i].load_state_dict(new_state_dict)
                elif self.args['model'] in ['dual-bert-grading']:
                    self.checkpointadapeter.init(
                        state_dict.keys(),
                        self.model.can_encoder.model.state_dict().keys(),
                    )
                    new_state_dict = self.checkpointadapeter.convert(state_dict)
                    self.model.can_encoder.model.load_state_dict(new_state_dict)
                    self.model.hard_can_encoder.model.load_state_dict(new_state_dict)
                elif self.args['model'] in ['dual-bert-proj']:
                    self.checkpointadapeter.init(
                        state_dict.keys(),
                        self.model.can_encoder.model.state_dict().keys(),
                    )
                    new_state_dict = self.checkpointadapeter.convert(state_dict)
                    self.model.can_encoder.model.load_state_dict(new_state_dict)
                else:
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
    
    @torch.no_grad()
    def test_model_fg(self, test_iter, print_output=False, rerank_agent=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        collection = {}
        for idx, batch in enumerate(pbar):                
            owner = batch['owner']
            label = batch['label']
            cid = batch['ids'].unsqueeze(0)
            cid_mask = torch.ones_like(cid)
            batch['ids'] = cid
            batch['ids_mask'] = cid_mask
            scores = self.model.predict(batch).cpu().tolist()    # [7]
            # print output
            if print_output:
                ctext = self.convert_to_text(batch['ids'].squeeze(0))
                self.log_save_file.write(f'[CTX] {ctext}\n')
                for rid, score in zip(batch['rids'], scores):
                    rtext = self.convert_to_text(rid)
                    score = round(score, 4)
                    self.log_save_file.write(f'[Score {score}] {rtext}\n')
                self.log_save_file.write('\n')
            if owner in collection:
                collection[owner].append((label, scores))
            else:
                collection[owner] = [(label, scores)]
        return collection
    
    
    @torch.no_grad()
    def test_model_horse_human(self, test_iter, print_output=False, rerank_agent=None):
        self.model.eval()
        pbar = tqdm(test_iter)
        collection = []
        for batch in pbar:                
            ctext = '\t'.join(batch['ctext'])
            rtext = batch['rtext']
            label = batch['label']

            cid = batch['ids'].unsqueeze(0)
            cid_mask = torch.ones_like(cid)
            batch['ids'] = cid
            batch['ids_mask'] = cid_mask

            scores = self.model.predict(batch).cpu().tolist()

            # print output
            if print_output:
                self.log_save_file.write(f'[CTX] {ctext}\n')
                assert len(rtext) == len(scores)
                for r, score, l in zip(rtext, scores, label):
                    score = round(score, 4)
                    self.log_save_file.write(f'[Score {score}, Label {l}] {r}\n')
                self.log_save_file.write('\n')

            collection.append((label, scores))
        return collection
    
    def train_model_adv(self, train_iter, test_iter, recoder=None, idx_=0, hard=False, whole_batch_num=0):
        self.model.train()
        total_loss, total_acc = 0, 0
        total_dc_acc = 0
        total_tloss, total_dc_loss = 0, 0
        total_tloss, total_bloss = 0, 0
        pbar = tqdm(train_iter)
        correct, s, oom_t = 0, 0, 0
        batch_num = 0
        for idx, batch in enumerate(pbar):
            # add the progress for adv training
            p = (whole_batch_num + batch_num) / self.args['total_step']
            l = 2. / (1. + np.exp(-10. * p)) - 1
            batch['l'] = l
            # 

            self.optimizer.zero_grad()
            with autocast():
                loss, dc_loss, acc, dc_acc = self.model(batch)
                tloss = loss + dc_loss
            self.scaler.scale(tloss).backward()
            self.scaler.unscale_(self.optimizer)
            clip_grad_norm_(self.model.parameters(), self.args['grad_clip'])
            self.scaler.step(self.optimizer)
            self.scaler.update()
            self.scheduler.step()

            total_tloss += tloss.item()
            total_loss += loss.item()
            total_dc_loss += dc_loss.item()
            total_acc += acc
            total_dc_acc += dc_acc
            batch_num += 1

            if whole_batch_num + batch_num in self.args['test_step']:
                self.test_now(test_iter, recoder)

            if recoder:
                recoder.add_scalar(f'train-epoch-{idx_}/Loss', total_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunLoss', loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/DCLoss', total_dc_loss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunDCLoss', dc_loss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/TotalLoss', total_tloss/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunTotalLoss', tloss.item(), idx)
                recoder.add_scalar(f'train-epoch-{idx_}/Acc', total_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunAcc', acc, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/DCAcc', total_dc_acc/batch_num, idx)
                recoder.add_scalar(f'train-epoch-{idx_}/RunDCAcc', dc_acc, idx)
             
            pbar.set_description(f'[!] loss: {round(loss.item(), 4)}|{round(total_loss/batch_num, 4)}; acc(acc|dc_acc): {round(total_acc/batch_num, 4)}|{round(total_dc_acc/batch_num, 4)}')

        if recoder:
            recoder.add_scalar(f'train-whole/Loss', total_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/DCLoss', total_dc_loss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/TotalLoss', total_tloss/batch_num, idx_)
            recoder.add_scalar(f'train-whole/Acc', total_acc/batch_num, idx_)
            recoder.add_scalar(f'train-whole/DCAcc', total_dc_acc/batch_num, idx_)
        return batch_num
