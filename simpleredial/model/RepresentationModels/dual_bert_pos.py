from model.utils import *

class BERTDualPositionWeightEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualPositionWeightEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask, cid_pos):
        # cid_pos: [B, S]
        cid_reps = self.ctx_encoder(cid, cid_mask)    # [B, S, E]
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep = (cid_pos.unsqueeze(-1) * cid_reps).sum(dim=1)    # [B, E]
        cid_rep /= cid_pos.sum(dim=-1).unsqueeze(-1)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask, cid_pos):
        cid_reps = self.ctx_encoder(ids, attn_mask)
        cid_rep = (cid_pos.unsqueeze(-1) * cid_reps).sum(dim=1)    # [B, E]
        cid_rep /= cid_pos.sum(dim=-1).unsqueeze(-1)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        cid_pos = batch['pos_w'].unsqueeze(0)

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, cid_pos)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        cid_pos = batch['pos_w']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, cid_pos)
        # distributed samples collected
        # cid_reps, rid_reps = distributed_collect(cid_rep, rid_rep)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B] or [B, 2*B]
        dot_product /= self.temp
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
