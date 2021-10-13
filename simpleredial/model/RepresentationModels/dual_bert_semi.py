from model.utils import *

class BERTDualSemiEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSemiEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)

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
        rid_rep = self.can_encoder(rid, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        ext_rid = batch['ext_rids']    # [N, E]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        ext_rid_mask = batch['ext_rids_mask']

        batch_size = cid.shape[0]
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, E]
        rid_rep_1 = self.can_encoder(rid, rid_mask)    # [B, E]
        rid_rep_2 = self.can_encoder(rid, rid_mask)    # [B, E] 
        ext_rid_rep_1 = self.can_encoder(ext_rid, ext_rid_mask)    # [N, E]
        ext_rid_rep_2 = self.can_encoder(ext_rid, ext_rid_mask)    # [N, E]

        # loss 1
        loss1 = 0
        dot_products = []
        for rid_rep, ext_rid_rep in zip([rid_rep_1, rid_rep_2], [ext_rid_rep_1, ext_rid_rep_2]):
            rid_rep = torch.cat([rid_rep, ext_rid_rep], dim=0)    # [B+N, E]
            dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B+N]
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1. 
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss1 += (-loss_.sum(dim=1)).mean()
            dot_products.append(dot_product.detach())
        loss1 /= 2

        # loss 2: unsupervised constrastive loss
        rid_rep_1 = torch.cat([rid_rep_1, ext_rid_rep_1], dim=0)    # [B+N, E]
        rid_rep_2 = torch.cat([rid_rep_2, ext_rid_rep_2], dim=0)    # [B+N, E]
        dot_product = torch.matmul(rid_rep_1, rid_rep_2.t())
        mask = torch.zeros_like(dot_product)
        mask[range(len(dot_product)), range(len(dot_product))] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss2 = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = 0
        for dp in dot_products:
            acc_num += (F.softmax(dp, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size / len(dot_products)

        # total loss
        loss = loss1 + loss2

        return loss, acc
