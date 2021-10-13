from model.utils import *

class BERTDualSAEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSAEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = SABertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, sid, tlid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, sid, tlid, cid_mask)
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
        sid = batch['sids'].unsqueeze(0)
        tlid = batch['tlids'].unsqueeze(0)
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, sid, tlid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product /= self.temp
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        sid = batch['sids']
        tlid = batch['tlids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, sid, tlid, rid, cid_mask, rid_mask)
        # gather all the embeddings in other processes
        # cid_rep, rid_rep = distributed_collect(cid_rep, rid_rep)

        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
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

        return loss, acc


class BERTDualSpeakerEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualSpeakerEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.ctx_encoder_1 = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.fusion_head = nn.Sequential(
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 768)
        )
        self.args = args

    def _encode(self, cid, sid, rid, cid_mask, sid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        sid_rep = self.ctx_encoder_1(sid, sid_mask)
        cid_rep = self.fusion_head(
            torch.cat([cid_rep, sid_rep], dim=-1)        
        )
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
        sid = batch['sids'].unsqueeze(0)
        cid_mask = torch.ones_like(cid)
        sid_mask = torch.ones_like(sid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, sid, rid, cid_mask, sid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product /= self.temp
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        sid = batch['sids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        sid_mask = batch['sids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, sid, rid, cid_mask, sid_mask, rid_mask)

        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
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

        return loss, acc
