from model.utils import *


class BERTDualCompEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualCompEncoder, self).__init__()
        model = args['pretrained_model']
        nhead = args['nhead']
        dim_feedforward = args['dim_ffd']
        dropout = args['dropout']
        num_encoder_layers = args['num_encoder_layers']
        self.topk = args['gray_cand_num'] + 1

        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        encoder_layer = nn.TransformerEncoderLayer(
            2*768, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
        )
        encoder_norm = nn.LayerNorm(2*768)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers, 
            encoder_norm,
        )
        self.trs_head = nn.Sequential(
            self.trs_encoder,
            nn.Dropout(p=dropout),
            nn.Tanh(),
            nn.Linear(768*2, 768)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(768*3, 768*2),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(768*2, 768),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 768)
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [b_c, E]
        rid_rep = self.can_encoder(rid, rid_mask)    # [b_r, E]

        b_c, b_r = len(cid_rep), len(rid_rep)
        cid_rep_ = cid_rep.unsqueeze(1).repeat(1, b_r, 1)    # [b_c, b_r, e]
        rid_rep_ = rid_rep.unsqueeze(0).repeat(b_c, 1, 1)    # [b_c, b_r, e]
        cross_rep = torch.cat([cid_rep_, rid_rep_], dim=-1)    # [b_c, b_r, 2*e]
        cross_rep = self.trs_head(cross_rep.permute(1, 0, 2)).permute(1, 0, 2)    # [b_r, b_c, 2*e] -> [b_c, b_r, e]
        reps = self.fusion_head(
            torch.cat([
                cid_rep_,    # [b_c, b_r, e]
                rid_rep_,    # [b_c, b_r, e]
                cross_rep,    # [b_c, b_r, e]
            ], dim=-1)
        )    # [b_c, b_r, e]
        cid_rep = cid_rep.unsqueeze(1)
        dp = torch.bmm(cid_rep, reps.permute(0, 2, 1)).squeeze(1)    # [b_c, b_r]
        return dp

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        dp = self._encode(cid, rid, cid_mask, rid_mask).squeeze(0)
        return dp
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        l1, l2 = len(cid), len(rid)
        dot_product = self._encode(cid, rid, cid_mask, rid_mask)
        
        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(l1), range(0, l2, self.topk)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, l2, self.topk)).cuda()).sum().item()
        acc = acc_num / l1
        return loss, acc
