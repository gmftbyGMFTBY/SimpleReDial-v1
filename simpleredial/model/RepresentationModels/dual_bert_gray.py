from model.utils import *

    
class BERTDualGrayEncoder(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, **args):
        super(BERTDualGrayEncoder, self).__init__()
        model = args['pretrained_model']
        self.gray_num = args['gray_cand_num']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

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
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid.unsqueeze(0), rid, None, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, 10*B]
        dot_product /= np.sqrt(768)

        mask = torch.zeros_like(dot_product).cuda()
        mask[torch.arange(batch_size), torch.arange(0, len(rid), self.gray_num+1)] = 1.
        # loss
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).sum().item()
        acc = acc_num / batch_size
        
        return loss, acc
