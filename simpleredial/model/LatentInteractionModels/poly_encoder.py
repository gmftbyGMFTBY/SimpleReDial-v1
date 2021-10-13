from model.utils import *


class PolyEncoder(nn.Module):
    
    def __init__(self, **args):
        super(PolyEncoder, self).__init__()
        model = args['pretrained_model']
        m = args['poly_m']
        self.can_encoder = BertEmbedding(model=model)
        self.ctx_encoder = TopKBertEmbedding(model=model, m=m)
        self.m = m
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        batch_size = len(cid)
        cid_rep = self.ctx_encoder(cid, cid_mask).permute(1, 0, 2)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cid_rep: [B_c, M, E]; rid_rep: [B_r, E]
        # [B_c, M, E] x [E, B_r] -> [B_c, M, B_r]-> [B_c, B_r, M]
        w_ = torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1)
        w_ /= np.sqrt(768)
        weights = F.softmax(w_, dim=-1)
        # [B_c, B_r, M] x [B_c, M, E] -> [B_c, B_r, E]
        cid_rep = torch.bmm(weights, cid_rep)
        rid_rep = rid_rep.unsqueeze(0).expand(batch_size, -1, -1)
        return cid_rep, rid_rep
        
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.einsum('ijk,ijk->ij', cid_rep, rid_rep).squeeze(0)    # [B_r]
        return dot_product
        
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # [B_c, B_r, E] x [B_c, B_r, E] -> [B_c, B_r]
        dot_product = torch.einsum('ijk,ijk->ij', cid_rep, rid_rep)
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()

        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc


class PolyEncoderHN(nn.Module):
    
    def __init__(self, **args):
        super(PolyEncoderHN, self).__init__()
        model = args['pretrained_model']
        m = args['poly_m']
        self.can_encoder = BertEmbedding(model=model)
        self.ctx_encoder = TopKBertEmbedding(model=model, m=m)
        self.m = m
        self.topk = args['gray_cand_num'] + 1
        
    def _encode(self, cid, rid, cid_mask, rid_mask):
        batch_size = len(cid)
        cid_rep = self.ctx_encoder(cid, cid_mask).permute(1, 0, 2)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cid_rep: [B_c, M, E]; rid_rep: [B_r, E]
        # [B_c, M, E] x [E, B_r] -> [B_c, M, B_r]-> [B_c, B_r, M]
        w_ = torch.matmul(cid_rep, rid_rep.t()).permute(0, 2, 1)
        w_ /= np.sqrt(768)
        weights = F.softmax(w_, dim=-1)
        # [B_c, B_r, M] x [B_c, M, E] -> [B_c, B_r, E]
        cid_rep = torch.bmm(weights, cid_rep)
        rid_rep = rid_rep.unsqueeze(0).expand(batch_size, -1, -1)
        return cid_rep, rid_rep
        
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.einsum('ijk,ijk->ij', cid_rep, rid_rep).squeeze(0)    # [B_r]
        return dot_product
        
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # [B_c, B_r, E] x [B_c, B_r, E] -> [B_c, B_r]
        dot_product = torch.einsum('ijk,ijk->ij', cid_rep, rid_rep)
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, len(rid), self.topk)] = 1.
        loss = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss.sum(dim=1)).mean()

        # calculate accuracy
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid),self.topk)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc
