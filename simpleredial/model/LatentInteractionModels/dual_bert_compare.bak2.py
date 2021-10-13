from model.utils import *

class BERTDualCompEncoder(nn.Module):

    '''This model needs the gray(hard negative) samples, which cannot be used for recall'''
    
    def __init__(self, **args):
        super(BERTDualCompEncoder, self).__init__()
        model = args['pretrained_model']
        s = args['smoothing']
        self.gray_num = args['gray_cand_num']
        nhead = args['nhead']
        dim_feedforward = args['dim_feedforward']
        dropout = args['dropout']
        num_encoder_layers = args['num_encoder_layers']
        self.bce_loss_scale = args['bce_loss_scale']

        # ====== Model ====== #
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        hidden_size = self.ctx_encoder.model.config.hidden_size
        encoder_layer = nn.TransformerEncoderLayer(
            hidden_size*2, 
            nhead=nhead, 
            dim_feedforward=dim_feedforward, 
            dropout=dropout,
        )
        encoder_norm = nn.LayerNorm(2*hidden_size)
        self.trs_encoder = nn.TransformerEncoder(
            encoder_layer,
            num_encoder_layers, 
            encoder_norm,
        )
        self.trs_head = nn.Sequential(
            self.trs_encoder,
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size*2, hidden_size)
        )
        self.gate = nn.Sequential(
            nn.Linear(hidden_size*3, hidden_size*2),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size*2, hidden_size),
            nn.Sigmoid()
        )

        self.cls_head = nn.Sequential(
            nn.Linear(hidden_size, hidden_size),
            nn.ReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(hidden_size, 1)
        )
        self.criterion = nn.BCEWithLogitsLoss()

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        b_c = len(cid_rep)
        rid_reps = rid_rep.unsqueeze(0).repeat(b_c, 1, 1)   # [B_c, B_r*gray, E]
        # fuse context into the response
        cid_reps = cid_rep.unsqueeze(1).repeat(1, len(rid), 1)    # [B_c, B_r*gray, E]
        for_comp = torch.cat([rid_reps, cid_reps], dim=-1)   # [B_c, B_r*gray, 2*E]
        comp_reps = self.trs_head(for_comp.permute(1, 0, 2)).permute(1, 0, 2)    # [B_c, B_r*gray, E]
        for_gate = torch.cat([comp_reps, rid_reps, cid_reps], dim=-1)
        gate_score = self.gate(for_gate)     # [B_c, B_r*gray, E]
        rid_fusion_reps = gate_score * rid_reps + (1-gate_score) * comp_reps
        # rid_reps: [B_c, G, E]; cid_rep: [B_c, E]
        rid_fusion_reps = rid_fusion_reps.permute(1, 0, 2)    # [G, B_c, E]

        # [B_c, E]; [G, B_c, E], [B_c, G, E]
        return cid_rep, rid_fusion_reps, comp_reps

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
        cid = cid.unsqueeze(0)
        cid_mask = torch.ones_like(cid)

        batch_size = rid.shape[0]
        cid_rep, rid_rep, _ = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.einsum('ijk,jk->ij', rid_rep, cid_rep).t()    # [B_c, G]
        dot_product /= np.sqrt(768)     # scale dot product
        return dot_product.squeeze(0)    # [G] = [B_r*gray]
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        b_c, b_r = len(cid), int(len(rid)//(self.gray_num+1))
        assert b_c == b_r
        # cid_rep: [B, E]; rid_rep: [G, B, E]; [B_c, G, E]
        cid_rep, rid_fusion_rep, comp_rid_rep = self._encode(cid, rid, cid_mask, rid_mask)

        # loss1: advantage loss
        comp_rid_rep = self.cls_head(comp_rid_rep).squeeze(-1)    # [B, G]
        label = torch.zeros_like(comp_rid_rep)
        label[torch.arange(b_c), torch.arange(0, len(rid), self.gray_num+1)] = 1.
        loss = self.bce_loss_scale * self.criterion(comp_rid_rep, label)

        # loss2: constrastive loss
        dot_product = torch.einsum('ijk,jk->ij', rid_fusion_rep, cid_rep).t()    # [B_c, G]
        dot_product /= np.sqrt(768)     # scale dot product

        mask = torch.zeros_like(dot_product).cuda()
        mask[torch.arange(b_c), torch.arange(0, len(rid), self.gray_num+1)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).sum().item()
        acc = acc_num / b_c

        return loss, acc
