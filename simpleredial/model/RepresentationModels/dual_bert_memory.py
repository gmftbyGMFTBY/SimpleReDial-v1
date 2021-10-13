from model.utils import *

class BERTDualMemoryEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualMemoryEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.alpha = args['memory_alpha']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.ctx_memory_block = nn.Sequential(
            nn.Linear(768, 768*2),
            nn.ReLU(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768),
        )
        self.res_memory_block = nn.Sequential(
            nn.Linear(768, 768*2),
            nn.ReLU(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768*2, 768),
        )
        self.memory_criterion = nn.MSELoss(reduction='mean')
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cosine similarity
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        # memory and normalization
        cid_rep_memory = self.ctx_memory_block(cid_rep)
        rid_rep_memory = self.res_memory_block(rid_rep)
        cid_rep_memory, rid_rep_memory = F.normalize(cid_rep_memory), F.normalize(rid_rep_memory)
        return cid_rep, rid_rep, cid_rep_memory, rid_rep_memory

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
        cid_rep, rid_rep, _, _ = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product *= self.temp
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep, cid_rep_memory, rid_rep_memory = self._encode(cid, rid, cid_mask, rid_mask)

        # constrastive loss
        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        dot_product /= self.temp
        batch_size = len(cid_rep)

        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        # memroy loss
        # loss += self.alpha * self.memory_criterion(cid_rep_memory, rid_rep)
        # loss += self.alpha * self.memory_criterion(rid_rep_memory, cid_rep)
        loss += self.alpha * self.memory_criterion(cid_rep_memory, rid_rep_memory)

        return loss, acc
