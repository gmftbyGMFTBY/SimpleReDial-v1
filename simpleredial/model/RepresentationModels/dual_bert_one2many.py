from model.utils import *

class BERTDualO2MEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualO2MEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoders = nn.ModuleList([
            BertEmbedding(model=model) for _ in range(self.topk) 
        ])

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_reps = []
        for cid_, cid_mask_ in zip(cid, cid_mask):
            cid_rep = self.ctx_encoder(cid_, cid_mask_)    # [B, E]
            cid_reps.append(cid_rep)
        rid_reps = []
        for i in range(self.topk):
            rid_rep = self.can_encoders[i](rid, rid_mask)
            rid_reps.append(rid_rep)
        return cid_reps, rid_reps

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_reps = []
        for idx in range(self.topk):
            rid_rep = self.can_encoders[idx](ids, attn_mask)
            rid_reps.append(rid_rep)
        # K*[B, E]
        rid_rep = torch.cat(rid_reps)    # [B*K, E]
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
        rid_rep_1 = self.can_encoders[0](rid, rid_mask)
        rid_rep_2 = self.can_encoders[1](rid, rid_mask)
        dot_products = []
        for rid_rep in [rid_rep_1, rid_rep_2]:
            dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)    # [B]
            dot_products.append(dot_product)
        dot_product = torch.stack(dot_products)    # [K, B]
        score = dot_product.max(dim=0)[0]
        return score

    def forward(self, batch):
        cid = batch['ids']     # 2*[B, S]
        rid = batch['rids']     # [B, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = len(rid)
        cid_reps, rid_reps = self._encode(cid, rid, cid_mask, rid_mask)
        dot_products = []
        loss = 0
        for cid_rep, rid_rep in zip(cid_reps, rid_reps):
            dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
            dot_product /= self.temp
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1. 
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
            dot_products.append(dot_product)
        loss /= 2
        # acc
        acc_num = 0
        for dp in dot_products:
            acc_num += (F.softmax(dp, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size / 2
        return loss, acc


class BERTDualO2MTopKEncoder(nn.Module):

    '''
    dual bert and dual latent interaction: one-to-many mechanism
    Top-k context embeddings and K candidates embeddings
    '''
    
    def __init__(self, **args):
        super(BERTDualO2MTopKEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.temp = args['temp']
        self.ctx_encoder = TopKBertEmbedding(model=model, m=self.topk)
        self.can_encoder = BertEmbedding(model=model)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [M+1, B, E]
        # [M+1, B, E]
        rid_rep = []
        for rid_, rid_mask_ in zip(rid, rid_mask):
            rid_rep_ = self.can_encoder(rid_, rid_mask_)    # [B, E]
            rid_rep.append(rid_rep_)
        rid_rep = torch.stack(rid_rep)    # [M+1, B, E]
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_reps = self.ctx_encoder(ids, attn_mask)     # [B, M, E]
        return cid_reps

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        # [M, 1, E]; [10, E]
        cid_reps, rid_reps = self._encode(cid, [rid], cid_mask, [rid_mask])
        rid_rep = rid_reps[0]
        # [M, 1, E] x [E, 10] -> [M, 1, 10]
        dot_product = torch.matmul(cid_reps, rid_rep.t()).squeeze(1)    # [M, 10]
        # score = dot_products.max(dim=0)[0]
        score = dot_product.mean(dim=0)
        return score
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = len(cid)
        # [M+1, B, E], [M+1, B, E]
        cid_reps, rid_reps = self._encode(cid, rid, cid_mask, rid_mask)

        loss = 0
        dot_products = []
        for cid_rep, rid_rep in zip(cid_reps, rid_reps):
            dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
            dot_product /= self.temp
            # constrastive loss
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1. 
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
            dot_products.append(dot_product)

        # acc
        acc_num = 0
        for dot_product_ in dot_products:
            acc_num += (F.softmax(dot_product_, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size / self.topk
        return loss, acc


class BERTDualO2M1Encoder(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, **args):
        super(BERTDualO2M1Encoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, E]
        rid_reps = []
        for rid_, rid_mask_ in zip(rid, rid_mask):
            rid_rep = self.can_encoder(rid_, rid_mask_)
            rid_reps.append(rid_rep)
        return cid_rep, rid_reps

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_reps = []
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
        cid_rep, rid_reps = self._encode(cid, [rid], cid_mask, [rid_mask])
        rid_rep = rid_reps[0]
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = len(cid)
        cid_rep, rid_reps = self._encode(cid, rid, cid_mask, rid_mask)
        loss = 0
        dot_products = []
        for i, rid_rep in enumerate(rid_reps):
            # cid_rep: [B, E]; rid_rep: [B, E]
            dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1. 
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
            dot_products.append(dot_product)
        loss /= len(rid_reps)

        # acc
        acc_num = 0
        for dot_product_ in dot_products:
            acc_num += (F.softmax(dot_product_, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size / len(dot_products)
        return loss, acc


class BERTDualO2MOriginalEncoder(nn.Module):

    '''dual bert and dual latent interaction: one-to-many mechanism'''
    
    def __init__(self, **args):
        super(BERTDualO2MOriginalEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['topk_encoder']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoders = nn.ModuleList([
            BertEmbedding(model=model) for _ in range(self.topk) 
        ])

    def _encode(self, cid, rid, cid_mask, rid_mask, test=False):
        cid_rep = self.ctx_encoder(cid, cid_mask)    # [B, E]
        rid_reps = []
        for idx in range(self.topk):
            rid_rep = self.can_encoders[idx](rid[idx], rid_mask[idx])
            rid_reps.append(rid_rep)
        return cid_rep, rid_reps

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_reps = []
        for idx in range(self.topk):
            rid_rep = self.can_encoders[idx](ids, attn_mask)
            rid_reps.append(rid_rep)
        # K*[B, E]
        rid_rep = torch.cat(rid_reps)    # [B*K, E]
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
        cid_rep, rid_reps = self._encode(cid, [rid] * self.topk, cid_mask, [rid_mask] * self.topk, test=True)
        dot_products = []
        for rid_rep in rid_reps:
            dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)    # [B]
            dot_products.append(dot_product)
        dot_products = torch.stack(dot_products)    # [K, B]
        score = dot_products.max(dim=0)[0]
        return score
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = len(cid)
        cid_rep, rid_reps = self._encode(cid, rid, cid_mask, rid_mask)
        loss = 0
        dot_products = []
        for i, rid_rep in enumerate(rid_reps):
            # cid_rep: [B, E]; rid_rep: [B, E]
            dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
            dot_product /= self.temp
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1. 
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss += (-loss_.sum(dim=1)).mean()
            dot_products.append(dot_product)
        loss /= len(rid_reps)
        
        # acc
        acc_num = 0
        for dot_product_ in dot_products:
            acc_num += (F.softmax(dot_product_, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size / self.topk
        return loss, acc
