from model.utils import *

class LatentInteractionAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(LatentInteractionAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
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
        self.show_parameters(self.args)
        
    def load_bert_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        self.model.ctx_encoder.load_bert_model(state_dict)
        self.model.can_encoder.load_bert_model(state_dict)
        print(f'[!] load pretrained BERT model from {path}')
        
    def train_model(self, train_iter, test_iter, recoder=None, idx_=0):
        self.model.train()
        total_loss, total_acc, batch_num = 0, 0, 0
        pbar = tqdm(train_iter)
        correct, s = 0, 0
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
    def test_model(self, test_iter, print_output=False, rerank_agent=None, core_time=False):
        self.model.eval()
        pbar = tqdm(test_iter)
        total_mrr, total_prec_at_one, total_map = 0, 0, 0
        total_examples, total_correct = 0, 0
        k_list = [1, 2, 5, 10]
        core_time_rest = 0
        for idx, batch in tqdm(list(enumerate(pbar))):                
            label = batch['label']
            batch['ids'] = batch['ids'].unsqueeze(0)
            batch['ids_mask'] = torch.ones_like(batch['ids'])
            if self.args['mode'] in ['train']:
                scores = self.model.module.predict(batch).cpu().tolist()    # [B]
            else:
                if core_time:
                    bt = time.time()
                scores = self.model.predict(batch).cpu().tolist()    # [B]
                if core_time:
                    core_time_rest += time.time() - bt
            # print output
            if print_output:
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
            total_examples += math.ceil(label.size()[0] / 10)
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
    
    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        if self.args['mode'] == 'train':
            # context encoder checkpoint
            self.checkpointadapeter.init(
                state_dict.keys(),
                self.model.ctx_encoder.model.state_dict().keys(),
            )
            new_state_dict = self.checkpointadapeter.convert(state_dict)
            self.model.ctx_encoder.model.load_state_dict(new_state_dict)
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
