from model.utils import *

class BERTDualMultiEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualMultiEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.topk = args['topk']
        self.num = args['gray_cand_num']
        self.vote_mode = args['vote_mode_for_test']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoders = nn.ModuleList([
            BertEmbedding(model=model, add_tokens=1)
            for _ in range(self.num)    
        ])
        self.args = args
    
    def _encode_topk(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        cid_rep = F.normalize(cid_rep)
        rid_reps = []
        random_idx = random.sample(range(self.num), self.topk)
        for i in random_idx:
            rid_rep = self.can_encoders[i](rid, rid_mask)
            rid_rep = F.normalize(rid_rep)
            rid_reps.append(rid_rep)
        return cid_rep, rid_reps

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        cid_rep = F.normalize(cid_rep)
        rid_reps = []
        for i in range(self.num):
            rid_rep = self.can_encoders[i](rid, rid_mask)
            rid_rep = F.normalize(rid_rep)
            rid_reps.append(rid_rep)
        return cid_rep, rid_reps

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep = self.ctx_encoder(cid, cid_mask)
        dp = []
        for i in range(self.num):
            rid_rep = self.can_encoders[i](rid, rid_mask)
            dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
            dp.append(dot_product)
        if self.vote_mode == 'mean':
            dp = torch.stack(dp).mean(dim=0)
        elif self.vote_mode == 'max':
            dp = torch.stack(dp).max(dim=0)
        return dp
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        # cid_rep, rid_reps = self._encode_topk(cid, rid, cid_mask, rid_mask)
        cid_rep, rid_reps = self._encode(cid, rid, cid_mask, rid_mask)

        acc_num, loss = 0, 0
        for i in range(len(rid_reps)):
            dot_product = torch.matmul(cid_rep, rid_reps[i].t()) 
            dot_product /= self.temp
            batch_size = len(cid_rep)
            
            # constrastive loss
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1. 
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
            
            # acc
            acc_num += (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size / len(rid_reps)
        return loss, acc
