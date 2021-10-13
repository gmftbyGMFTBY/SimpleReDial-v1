from model.utils import *

class BERTDualMixUpEncoder(nn.Module):

    '''Sentence-level MixUp data augmentation technique is used'''

    def __init__(self, **args):
        super(BERTDualMixUpEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbeddingWithWordEmbd(model=model, add_tokens=1)
        self.alpha = args['alpha']

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep, rid_we = self.can_encoder(rid, rid_mask)    # [B, E]; [B, S, E]
        lam = np.random.beta(self.alpha, self.alpha)    # [B, E]
        index = torch.randperm(len(cid))
        # x, y: [B, E]
        mixed_y = lam * rid_we + (1 - lam) * rid_we[index, :, :]
        rid_mix_rep = self.can_encoder(rid, rid_mask, word_embeddings=mixed_y)
        return cid_rep, rid_rep, rid_mix_rep

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
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0] * 2
        cid_rep, rid_rep, rid_mix_rep = self._encode(cid, rid, cid_mask, rid_mask)
        rid_rep = torch.cat([rid_rep, rid_mix_rep], dim=0)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
