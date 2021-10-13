from model.utils import *

class BERTDualCLEncoder(nn.Module):

    '''Contrastive Loss for Multi-turn conversation context'''

    def __init__(self, **args):
        super(BERTDualCLEncoder, self).__init__()
        model = args['pretrained_model']
        self.large_temp = args['large_temp']
        self.mini_temp = args['mini_temp']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cosine similarity
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        return cid_rep, rid_rep

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
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
   
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        # pos/neg: [B, S]
        pos_cid = batch['pos_ids']
        pos_cid_mask = batch['pos_ids_mask']
        neg_cid = batch['neg_ids']
        neg_cid_mask = batch['neg_ids_mask']
        batch_size = len(cid)

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        pos_cid_rep = self.ctx_encoder(pos_cid, pos_cid_mask)
        neg_cid_rep = self.ctx_encoder(neg_cid, neg_cid_mask)
        pos_cid_rep, neg_cid_rep = F.normalize(pos_cid_rep), F.normalize(neg_cid_rep)

        # context-response contrastive loss and basic accuracy for monitor
        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        dot_product /= self.large_temp
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        # context-context contrastive loss
        # cosine similarity between anchor and positive and negative
        pos_dp = torch.einsum('ij,ij->i', cid_rep, pos_cid_rep)    # [B]
        neg_dp = torch.einsum('ij,ij->i', cid_rep, neg_cid_rep)    # [B]
        dp = torch.stack([pos_dp, neg_dp]).t()    # [B, 2]
        # divide the mini temp for this hard negative optimizing
        dp /= self.mini_temp

        mask = torch.zeros_like(dp)
        mask[:, 0] = 1. 
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        acc_num = (dp.max(dim=-1)[1] == torch.LongTensor([0] * len(dp)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc
