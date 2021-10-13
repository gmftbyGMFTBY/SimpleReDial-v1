from model.utils import *

class BERTDualExtNegEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualExtNegEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ext_neg_size = args['ext_neg_size']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
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
        dot_product /= self.temp
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        # [B, E], [B+M, E]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # [B, B+M]
        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        dot_product /= np.sqrt(768)
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
        
        # loss 2
        dot_product = torch.matmul(rid_rep, rid_rep.t()) 
        dot_product /= np.sqrt(768)
        dot_product /= self.temp
        batch_size = len(rid_rep)
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        return loss, acc
