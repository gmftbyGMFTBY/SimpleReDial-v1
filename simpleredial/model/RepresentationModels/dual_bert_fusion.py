from model.utils import *

class BERTDualFusionEncoder(nn.Module):

    '''fusion the context information into the response representation to improve the recall performance:
    However, the very long context will bring the noise into the response representation. Thus, the response information is used to select the useful information in the conversation context and fuse them into the final response representations.
    '''

    def __init__(self, **args):
        super(BERTDualFusionEncoder, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']
        p = args['dropout']
        self.ctx_encoder = BertFullEmbedding(model=model)
        self.can_encoder = BertFullEmbedding(model=model)
        self.label_smooth_loss = LabelSmoothLoss(smoothing=s)

        # Gated module
        self.select_gate = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.ReLU(),
            nn.Dropout(p=p),
            nn.Linear(768, 768),
            nn.Sigmoid(),
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        rid_rep = rid_rep[:, 0, :]    # [B, E]
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, cids, cid_attn_mask, ids, attn_mask):
        bsz = len(cids)
        cid_rep, rid_rep = self._encode(cids, ids, cid_attn_mask, attn_mask)
        ext_cid_rep = self.extract_from_context(rid_rep, cid_rep, cid_rep, cid_attn_mask)    # [B_r, B_c, E]
        rid_rep = rid_rep.unsqueeze(1).repeat(1, bsz, 1)    # [B_r, B_c, E]

        gate_score = self.select_gate(
            torch.cat([ext_cid_rep, rid_rep], dim=-1)        
        )     # [B_r, B_c, E]
        rid_rep = gate_score * ext_cid_rep + (1 - gate_score) * rid_rep    # [B_r, B_c, E]
        rid_rep = rid_rep[torch.arange(bsz), torch.arange(bsz), :]    # [B_r, E]
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)[:, 0, :]    # [E]
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        cid = cid.unsqueeze(0)    # [1, S]
        cid_mask = torch.ones_like(cid)    # [1, S]

        b_c, b_r = len(cid), len(rid)

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        ext_cid_rep = self.extract_from_context(rid_rep, cid_rep, cid_rep, cid_mask)    # [B_r, 1, E]
        rid_rep = rid_rep.unsqueeze(1).repeat(1, b_c, 1)    # [B_r, 1, E]

        gate_score = self.select_gate(
            torch.cat([ext_cid_rep, rid_rep], dim=-1)        
        )     # [B_r, 1, E]
        rid_rep = gate_score * ext_cid_rep + (1 - gate_score) * rid_rep    # [B_r, 1, E]

        # rid: [B_r, 1, E]; cid: [1, E]
        dot_product = torch.einsum('ijk,jk->ij', rid_rep, cid_rep[:, 0, :]).squeeze(0)   # [B_r]
        dot_product /= np.sqrt(768)     # scale dot product
        return dot_product.squeeze(1)

    def extract_from_context(self, query, key, value, padding_mask):
        # query(from response): [B_r, E]; value/key(from context): [B_c, S, E]
        # padding_mask: [B_c, S]
        # return the tensor [B_r, B_c, E]
        b_c, b_r = len(key), len(query)
        weights = torch.bmm(
            query.unsqueeze(0).repeat(b_c, 1, 1),     # [B_c, B_r, E]
            key.permute(0, 2, 1),      # [B_c, E, S]
        ).permute(1, 0, 2)    # [B_r, B_c, S]
        weights /= np.sqrt(768)
        mask_ = torch.where(padding_mask != 0, torch.zeros_like(padding_mask), torch.ones_like(padding_mask))
        mask_ = mask_ * -1e3
        mask_ = mask_.unsqueeze(0).repeat(b_r, 1, 1)    # [B_r, B_c, S]
        weights += mask_    # [B_r, B_c, S]
        weights = F.softmax(weights, dim=-1).permute(1, 0, 2)    # [B_c, B_r, S]
        rep = torch.bmm(weights, value).permute(1, 0, 2)    # [B_r, B_c, E]
        return rep
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        b_c, b_r = len(cid), len(rid)

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        ext_cid_rep = self.extract_from_context(rid_rep, cid_rep, cid_rep, cid_mask)    # [B_r, B_c, E]
        rid_rep = rid_rep.unsqueeze(1).repeat(1, b_c, 1)    # [B_r, B_c, E]

        gate_score = self.select_gate(
            torch.cat([ext_cid_rep, rid_rep], dim=-1)        
        )     # [B_r, B_c, E]
        rid_rep = gate_score * ext_cid_rep + (1 - gate_score) * rid_rep    # [B_r, B_c, E]

        # rid: [B_r, B_c, E]; cid: [B_c, E] => [B_r, B_c]
        dot_product = torch.einsum('ijk,jk->ij', rid_rep, cid_rep[:, 0, :]).t()    # [B_c, B_r]
        dot_product /= np.sqrt(768)     # scale dot product

        # label smooth loss
        gold = torch.arange(b_c).cuda()
        loss = self.label_smooth_loss(dot_product, gold)

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(b_c)).cuda()).sum().item()
        acc = acc_num / b_c

        return loss, acc
