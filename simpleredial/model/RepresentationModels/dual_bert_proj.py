from model.utils import *

class BERTDualHNProjEncoder(nn.Module):

    '''With additional projection'''

    def __init__(self, **args):
        super(BERTDualHNProjEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        p = args['dropout']
        m = args['poly_m']
        self.ctx_encoder = TopKBertEmbedding(model=model, m=m, dropout=p)
        self.can_encoder = TopKBertEmbedding(model=model, m=m, dropout=p)
        proj_size = args['proj_size']
        ln_eps = float(args['layer_norm_eps'])
        self.ctx_proj = nn.Sequential(
            nn.Linear(768*m, 768*m),
            nn.Tanh(),
            nn.Dropout(p=p),
            nn.Linear(768*m, proj_size),
        )
        self.res_proj = nn.Sequential(
            nn.Linear(768*m, 768*m),
            nn.Tanh(),
            nn.Dropout(p=p),
            nn.Linear(768*m, proj_size),
        )
        self.ctx_layer_norm = nn.LayerNorm(proj_size, eps=ln_eps)
        self.res_layer_norm = nn.LayerNorm(proj_size, eps=ln_eps)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # [M, B, E] -> [B, M*E]
        cid_rep = cid_rep.permute(1, 0, 2)
        cid_rep = cid_rep.reshape(len(cid_rep), -1)
        rid_rep = rid_rep.permute(1, 0, 2)
        rid_rep = rid_rep.reshape(len(rid_rep), -1)
        cid_rep, rid_rep = self.ctx_proj(cid_rep), self.res_proj(rid_rep)
        cid_rep = self.ctx_layer_norm(cid_rep)
        rid_rep = self.res_layer_norm(rid_rep)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        rid_rep = rid_rep.permute(1, 0, 2)
        rid_rep = rid_rep.reshape(len(rid_rep), -1)
        rid_rep = self.res_proj(rid_rep)
        rid_rep = self.res_layer_norm(rid_rep)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        cid_rep = cid_rep.permute(1, 0, 2)
        cid_rep = cid_rep.reshape(len(cid_rep), -1)
        cid_rep = self.ctx_proj(cid_rep)
        cid_rep = self.ctx_layer_norm(cid_rep)
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
        rid = batch['rids']    # [B*M, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, len(rid), self.topk)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.topk)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
