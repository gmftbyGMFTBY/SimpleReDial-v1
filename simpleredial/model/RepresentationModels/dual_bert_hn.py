from model.utils import *

class BERTDualHNEncoder(nn.Module):

    '''Dual bert with hard negative samples'''

    def __init__(self, **args):
        super(BERTDualHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        # self.easy_margin = args['easy_margin']
        # self.hard_margin = args['hard_margin']
        self.temp = args['temp']
        # self.easy_criterion = nn.TripletMarginWithDistanceLoss(
        #     margin=self.easy_margin, 
        #     reduction='sum',
        #     distance_function=cosine_distance,    
        # )
        # self.hard_criterion = nn.TripletMarginWithDistanceLoss(
        #     margin=self.hard_margin, 
        #     reduction='sum',
        #     distance_function=cosine_distance,    
        # )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
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
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # contrastive loss
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        dot_product /= self.temp
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, len(rid_rep), self.topk)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # get acc
        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.topk)).cuda()).sum().item()
        acc = acc_num / batch_size
            
        # margin loss for topk hard negative samples
        # hard_anchor_reps, hard_rid_reps, hard_nid_reps = [], [], []
        # easy_anchor_reps, easy_rid_reps, easy_nid_reps = [], [], []
        # for idx in range(batch_size):
        #     # hard
        #     for i in range(idx*self.topk+1, idx*self.topk+self.topk):
        #         hard_anchor_reps.append(cid_rep[idx])
        #         hard_rid_reps.append(rid_rep[idx*self.topk])
        #         hard_nid_reps.append(rid_rep[i])
        #     # easy
        #     index = list(set(range(len(rid))) - set(range(idx*self.topk, idx*self.topk + self.topk)))
        #     index = random.sample(index, self.topk)
        #     for i in index:
        #         easy_anchor_reps.append(cid_rep[idx])
        #         easy_rid_reps.append(rid_rep[idx*self.topk])
        #         easy_nid_reps.append(rid_rep[i])
        # easy_anchor_reps = torch.stack(easy_anchor_reps)
        # easy_rid_reps = torch.stack(easy_rid_reps)
        # easy_nid_reps = torch.stack(easy_nid_reps)
        # hard_anchor_reps = torch.stack(hard_anchor_reps)
        # hard_rid_reps = torch.stack(hard_rid_reps)
        # hard_nid_reps = torch.stack(hard_nid_reps)
        # loss += self.hard_criterion(hard_anchor_reps, hard_nid_reps, hard_rid_reps)
        # loss += self.easy_criterion(easy_anchor_reps, easy_nid_reps, easy_rid_reps)
        return loss, acc


class BERTDualHNHierEncoder(nn.Module):

    '''Dual bert with hard negative samples'''

    def __init__(self, **args):
        super(BERTDualHNHierEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        p = args['dropout']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.fg_head_ctx = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=p),
            nn.Linear(768, 768)
        )
        self.fg_head_res = nn.Sequential(
            nn.Linear(768, 768),
            nn.Tanh(),
            nn.Dropout(p=p),
            nn.Linear(768, 768)
        )
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        rid_rep_ = self.fg_head_res(rid_rep)
        rid_rep = torch.cat([rid_rep, rid_rep_], dim=1)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        cid_rep_ = self.fg_head_res(cid_rep)
        cid_rep = torch.cat([cid_rep, cid_rep_], dim=1)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cid_rep_ = self.fg_head_ctx(cid_rep)
        rid_rep_ = self.fg_head_res(rid_rep)
        cid_rep = torch.cat([cid_rep, cid_rep_], dim=1)
        rid_rep = torch.cat([rid_rep, rid_rep_], dim=1)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        
        # ====== ===== #
        rids = torch.stack(torch.split(rid, self.topk))    # [B, M, S]
        rids_mask = torch.stack(torch.split(rid_mask, self.topk))    # [B, M, S]
        rid, rid_mask = rids[:, 0, :], rids_mask[:, 0, :]    # [B, S]
        hrid, hrid_mask = rids[:, 1:, :], rids_mask[:, 1:, :]
        hrid, hrid_mask = hrid.reshape(-1, hrid.size(-1)), hrid_mask.reshape(-1, hrid_mask.size(-1))    # [B*M, S]
        # ====== ===== #

        # first layer: easy
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        batch_size = len(cid_rep)
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # second layer: hard
        cid_rep = self.fg_head_ctx(cid_rep)    # [B, E]
        rid_rep = self.fg_head_res(rid_rep)    # [B, E]
        hrid_rep = self.can_encoder(hrid, hrid_mask)    # [B*M, E]
        hrid_rep = self.fg_head_res(hrid_rep)    # [B*M, E]
        hrid_rep = torch.stack(torch.split(hrid_rep, self.topk-1))    # [B, M, E]
        hrid_rep = torch.cat([rid_rep.unsqueeze(1), hrid_rep], dim=1)    # [B, M+1, E]
        dp = torch.bmm(cid_rep.unsqueeze(1), hrid_rep.permute(0, 2, 1)).squeeze(1)    # [B, M+1]
        mask = torch.zeros_like(dp)
        mask[range(batch_size), 0] = 1.
        loss_ = F.log_softmax(dp, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.arange(batch_size).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc


class BERTDualHNPosEncoder(nn.Module):

    '''Dual bert with hard negative samples, and the position weight for the context'''

    def __init__(self, **args):
        super(BERTDualHNPosEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask, cid_pos):
        cid_reps = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep = (cid_pos.unsqueeze(-1) * cid_reps).sum(dim=1)
        cid_rep /= cid_pos.sum(dim=-1).unsqueeze(-1)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask, cid_pos):
        cid_reps = self.ctx_encoder(ids, attn_mask)
        cid_rep = (cid_pos.unsqueeze(-1) * cid_reps).sum(dim=1)
        cid_rep /= cid_pos.sum(dim=-1).unsqueeze(-1)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        cid_pos = batch['pos_w'].unsqueeze(0)

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, cid_pos)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B*M, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        cid_pos = batch['pos_w']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, cid_pos) 
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, len(rid_rep), self.topk)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid_rep), self.topk)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualSAHNEncoder(nn.Module):

    '''speaker-aware dual bert with hard negative samples'''

    def __init__(self, **args):
        super(BERTDualSAHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertFullEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.speaker_embedding = nn.Embedding(2, 768)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask, sids):
        # scids: [B, S];
        scid_rep = self.speaker_embedding(sids)    # [B, S, E]
        cid_rep = self.ctx_encoder(cid, cid_mask)   # [B, S, E] 
        rid_rep = self.can_encoder(rid, rid_mask)   # [B, E] 
        cid_rep += scid_rep
        # filter out the padding tokens
        cid_rep = cid_mask.unsqueeze(-1) * cid_rep    # [B, S, E]
        cid_rep = cid_rep.mean(dim=1)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask, sids):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        scid_rep = self.speaker_embedding(sids)
        cid_rep += scid_rep
        # filter out the padding tokens
        cid_rep = (cid_mask.unsqueeze(-1) * cid_rep).mean(dim=1)    # [B, E]
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']
        sid = batch['sids']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, sid)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B*M, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        sids = batch['sids']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, sids)
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


class BERTDualGradingEncoder(nn.Module):

    '''Dual bert with multi response encoder for hard negative samples'''

    def __init__(self, **args):
        super(BERTDualGradingEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.hard_can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.temp = args['temp']
        self.lamb = args['lambda']

    def _encode(self, cid, rid, hrid, cid_mask, rid_mask, hrid_mask):
        # cid/rid: [B, S]; hrid: [B, K, S], where K is the number of the hard negative samples
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)

        hrid, hrid_mask = hrid.reshape(-1, hrid.size(-1)), hrid_mask.reshape(-1, hrid_mask.size(-1))
        hrid_rep = self.hard_can_encoder(hrid, hrid_mask)    # [B*K, E]
        hrid_rep = torch.stack(torch.split(hrid_rep, self.topk))    # [B, K, E]
        return cid_rep, rid_rep, hrid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        hrid_rep = self.hard_can_encoder(ids, attn_mask)
        rid_rep = torch.cat([rid_rep, hrid_rep], dim=-1)    # [B, E*2]
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        cid_rep = torch.cat([cid_rep, cid_rep], dim=-1)    # [B, 2*E]
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
        hrid_rep = self.hard_can_encoder(rid, rid_mask)
        dp1 = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dp2 = torch.matmul(cid_rep, hrid_rep.t()).squeeze(0)
        dp = self.lamb * dp1 + (1-self.lamb) * dp2
        return dp
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B*M, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']    # [B, S]

        # ====== ===== #
        rids = torch.stack(torch.split(rid, self.topk))    # [B, M, S]
        rids_mask = torch.stack(torch.split(rid_mask, self.topk))    # [B, M, S]
        rid, rid_mask = rids[:, 0, :], rids_mask[:, 0, :]
        hrid, hrid_mask = rids, rids_mask
        # ====== ===== #

        batch_size = len(cid)

        cid_rep, rid_rep, hrid_rep = self._encode(cid, rid, hrid, cid_mask, rid_mask, hrid_mask)

        # 1. 
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        dot_product /= self.temp
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # 2. hrid_rep: [B, K, E] 
        # cid_rep: [B, 1, E] x [B, E, K+1]
        hrid_rep = hrid_rep.permute(0, 2, 1)    # [B, E, K+1]
        dot_product_2 = torch.bmm(cid_rep.unsqueeze(1), hrid_rep).squeeze(1)    # [B, K+1]
        mask = torch.zeros_like(dot_product_2)
        mask[range(batch_size), 0] = 1.
        loss_ = F.log_softmax(dot_product_2, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualHNEncoderBCE(nn.Module):

    '''Dual bert with hard negative samples, but the bce loss is used instead of the contrastive loss'''

    def __init__(self, **args):
        super(BERTDualHNEncoderBCE, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.criterion = nn.BCEWithLogitsLoss()
        self.fusion_head = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Linear(768, 1)
        )
        self.neg_num = args['neg_num']

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        bsz_c, bsz_r = len(cid_rep), len(rid_rep)
        cid_rep = cid_rep.unsqueeze(1).expand(-1, bsz_r, -1)    # [B_c, B_r, E]
        rid_rep = rid_rep.unsqueeze(0).expand(bsz_c, -1, -1)    # [B_c, B_r, E]
        rep = self.fusion_head(torch.cat([cid_rep, rid_rep], dim=-1)).squeeze(-1)    # [B_c, B_r]
        return rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        dot_product = self._encode(cid, rid, cid_mask, rid_mask).squeeze(0)    # [B_r]
        dot_product = torch.sigmoid(dot_product)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        dot_product = self._encode(cid, rid, cid_mask, rid_mask)
        batch_size = len(cid)

        # BCE loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, len(rid), self.topk)] = 1. 
        dot_product, mask = dot_product.view(-1), mask.view(-1)
        a, b = [], []
        for i in range(batch_size):
            index = list(range(self.topk*i, self.topk*i+self.topk))
            neg_index = random.sample(list(set(range(len(rid))) - set(index)), self.neg_num)
            index.extend(neg_index)
            for idx in index:
                a.append(dot_product[idx])
                b.append(mask[idx])
        a, b = torch.stack(a), torch.stack(b)
        a = dot_product.view(-1)
        b = mask.view(-1)

        # random shuffle
        random_idx = list(range(len(a)))
        random.shuffle(random_idx)
        a = torch.stack([a[i] for i in random_idx])
        b = torch.stack([b[i] for i in random_idx])
        loss = self.criterion(a, b)

        # acc
        acc = ((torch.sigmoid(dot_product).view(-1) > 0.5).float() == mask.view(-1)).float().mean().item()

        return loss, acc


class BERTDualHN2Encoder(nn.Module):

    '''Dual bert with hard negative samples:
        directly load the dual-bert model checkpoint trained on easy negative samples; and the BERT model will not be updated.'''

    def __init__(self, **args):
        super(BERTDualHN2Encoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.temp = args['temp']

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
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product /= self.temp
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']    # [B*M, S]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        dot_product /= self.temp
        batch_size = len(cid_rep)

        # mask the others easy hard negative (for current context, they are easy negative samples)
        nset = set(range(0, len(rid), self.topk))
        for i in range(batch_size):
            nnset = nset | set(range(self.topk*i, self.topk*i+self.topk))
            index = [i for i in range(len(rid)) if i not in nnset]
        #     index = [i for i in range(len(rid)) if i not in nset]
            dot_product[i, index] = -1e3
        #
        # mask hard negative 
        # for i in range(batch_size):
        #     nnset = set(range(self.topk*i+1, self.topk*i+self.topk))
        #     index = [i for i in range(len(rid)) if i in nnset]
        #     dot_product_old[i, index] = -1e3
        # mask easy negative, only hard negative and ground-truth are used
        # for i in range(batch_size):
        #     nset = set(range(self.topk*i, self.topk*i+self.topk))
        #     index = [i for i in range(len(rid)) if i not in nset]
        #     dot_product[i, index] = -1e3
        # 
        # mask based on the margin
        # mt_label = - dot_product + dot_product[range(batch_size), range(0, len(rid), self.topk)].unsqueeze(-1)   # True are masked
        # mt_label = mt_label > self.margin_threshold
        # dot_product[mt_label] = -1e3

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(0, len(rid), self.topk)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.topk)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualHNCTXEncoder(nn.Module):

    '''Dual bert with hard negative samples for context'''

    def __init__(self, **args):
        super(BERTDualHNCTXEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 2    # gray hard negative and 2 positive
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.temp = args['temp']

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
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        dot_product /= self.temp
        return dot_product
    
    def get_cl_loss_from_dot_product(self, dot_product):
        # constrastive loss
        dot_product /= np.sqrt(768)
        batch_size = len(dot_product)
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        return loss
    
    def forward(self, batch):
        cid = batch['ids']     # [B, S]
        cid1 = batch['ids1']    # [B, S]
        cid2 = batch['ids2']    # [B*M, S]
        rid = batch['rids']    # [B, S]
        cid_mask = batch['ids_mask']
        cid1_mask = batch['ids1_mask']
        cid2_mask = batch['ids2_mask']
        rid_mask = batch['rids_mask']

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)    # [B, E]
        cid1_rep = self.ctx_encoder(cid1, cid1_mask)    # [B, E]
        cid2_rep = self.ctx_encoder(cid2, cid2_mask)    # [B*M, E]

        # pos context 1
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]
        dot_product /= self.temp
        loss = self.get_cl_loss_from_dot_product(dot_product)
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(len(dot_product))).cuda()).sum().item()
        acc = acc_num / len(dot_product)
        
        # pos context 2
        dot_product = torch.matmul(cid1_rep, rid_rep.t())     # [B, B]
        dot_product /= self.temp
        loss += self.get_cl_loss_from_dot_product(dot_product)

        # hard negative for positive context
        ctx_dp = torch.matmul(cid_rep, cid1_rep.t())     # [B, B]
        ctx_dp_2 = torch.matmul(cid_rep, cid2_rep.t())   # [B, B*M]
        ctx_dp = torch.cat([ctx_dp, ctx_dp_2], dim=-1)    # [B, B+B*M]
        loss += self.get_cl_loss_from_dot_product(ctx_dp)
        
        return loss, acc


class BERTDualShuffleCtxHNEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualShuffleCtxHNEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        # self.temp = args['temp']

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
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        # dot_product /= self.temp
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        nid = batch['nids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        nid_mask = batch['nids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        cid_rep_ = self.ctx_encoder(cid, cid_mask)
        nid_rep = self.ctx_encoder(nid, nid_mask)
        nid_rep = torch.cat([cid_rep_, nid_rep], dim=0)    # [2*B, E]

        # c-c contrastive loss
        dot_product = torch.matmul(cid_rep, nid_rep.t())     # [B, 2*B]
        dot_product /= np.sqrt(768)
        # dot_product /= self.temp
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # constrastive loss
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B*K]
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss += (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc


class BERTDualHNCurriculumEncoder(nn.Module):

    '''Que2Search: Fast and Accurate Query and Document Understanding for Search at Facebook'''

    def __init__(self, **args):
        super(BERTDualHNCurriculumEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num'] + 1
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.margin = args['margin']
        self.temp = args['temp']
        self.criterion = nn.TripletMarginWithDistanceLoss(
            margin=0.1, 
            reduction='sum',
            distance_function=cosine_distance,    
        )

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
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
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        batch_size = len(cid)
        
        # get the matching matrix
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())

        if batch['mode'] == 'easy':
            # constrastive loss
            # only the contrastive loss needs the temperature
            dot_product /= self.temp
            mask = torch.zeros_like(dot_product)
            mask[range(batch_size), range(batch_size)] = 1. 
            loss_ = F.log_softmax(dot_product, dim=-1) * mask
            loss = (-loss_.sum(dim=1)).mean()
            # get acc
            acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
            acc = acc_num / batch_size
        else:
            # margin loss for topk hard negative samples
            anchor_reps, rid_reps, nid_reps = [], [], []
            for idx in range(batch_size):
                for i in range(idx*self.topk+1, idx*self.topk+self.topk):
                    anchor_reps.append(cid_rep[idx])
                    rid_reps.append(rid_rep[idx*self.topk])
                    nid_reps.append(rid_rep[i])
            anchor_reps = torch.stack(anchor_reps)
            rid_reps = torch.stack(rid_reps)
            nid_reps = torch.stack(nid_reps)
            loss = self.criterion(anchor_reps, nid_reps, rid_reps)
            # get acc
            acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.topk)).cuda()).sum().item()
            acc = acc_num / batch_size
        return loss, acc


class BERTDualHNTripletMarginLossEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHNTripletMarginLossEncoder, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.margin = args['margin']
        self.hard_margin = args['hard_margin']

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
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
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product

    def _fast_triplet_margin_cosine_distance_loss(self, cid_rep, rid_rep):
        cosine_sim = torch.matmul(cid_rep, rid_rep.t())
        cosine_sim = (1 + cosine_sim) / 2
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        hrid = batch['hrids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        hrid_mask = batch['hrids_mask']
        batch_size = len(cid)

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
            
        # margin loss for topk hard negative samples
        ipdb.set_trace()
        anchor_reps, rid_reps, nid_reps = [], [], []
        for idx in range(batch_size):
            for i in range(idx*self.topk+1, idx*self.topk+self.topk):
                anchor_reps.append(cid_rep[idx])
                rid_reps.append(rid_rep[idx*self.topk])
                nid_reps.append(rid_rep[i])
        anchor_reps = torch.stack(anchor_reps)
        rid_reps = torch.stack(rid_reps)
        nid_reps = torch.stack(nid_reps)
        loss = self.criterion(anchor_reps, nid_reps, rid_reps)
        # get acc
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B*K]
        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.topk)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc
