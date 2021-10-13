from model.utils import *


class BERTDualEncoderTripletMargin(nn.Module):

    def __init__(self, **args):
        super(BERTDualEncoderTripletMargin, self).__init__()
        model = args['pretrained_model']
        self.topk = args['gray_cand_num']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args
        self.easy_margin = args['easy_margin']
        self.hard_margin = args['hard_margin']
        self.easy_criterion = nn.TripletMarginWithDistanceLoss(
            margin=self.easy_margin,
            reduction='sum',
            distance_function=cosine_distance,
        )
        self.hard_criterion = nn.TripletMarginWithDistanceLoss(
            margin=self.hard_margin,
            reduction='sum',
            distance_function=cosine_distance,
        )

    def _fast_triplet_margin_cosine_distance_loss(self, cid_rep, rid_rep):
        '''cid_rep/rid_rep: [B, E]; reduction is the `sum`'''
        cosine_sim = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        cosine_sim = (1 + cosine_sim) / 2    # nonnegative real-value number, range from 0 to 1
        # triplet with margin loss
        cosine_sim = self.margin + cosine_sim - cosine_sim.diag().unsqueeze(1)
        cosine_sim = torch.where(cosine_sim > 0, cosine_sim, torch.zeros_like(cosine_sim))
        # ignore the ground-truth
        cosine_sim[range(len(cid_rep)), range(len(cid_rep))] = 0.
        # only topk negative will be optimized
        values = torch.topk(cosine_sim, self.topk, dim=-1)[0]    # [B, K]
        loss = values.sum()
        return loss

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cosine similarity needs the normalization
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        return cid_rep, rid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)    # [B]
        return dot_product
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        hrid = batch['hrids']    # [K*B, E]
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        hrid_mask = batch['hrids_mask']
        batch_size = len(cid)

        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        hrid_rep = self.can_encoder(hrid, hrid_mask)
        hrid_rep = F.normalize(hrid_rep)
        
        # get acc
        dot_product = torch.matmul(cid_rep, rid_rep.t())
        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        # margin loss for topk easy negative samples
        dot_product[range(batch_size), range(batch_size)] = -1e3
        hn_index = torch.topk(dot_product, self.topk, dim=-1)[1].tolist()
        anchor_reps, rid_reps, nid_reps = [], [], []
        for idx in range(batch_size):
            hn = hn_index[idx]
            for i in hn:
                anchor_reps.append(cid_rep[idx])
                rid_reps.append(rid_rep[idx])
                nid_reps.append(rid_rep[i])
        anchor_reps = torch.stack(anchor_reps)
        rid_reps = torch.stack(rid_reps)
        nid_reps = torch.stack(nid_reps)
        loss = self.easy_criterion(anchor_reps, nid_reps, rid_reps)

        # margin loss for topk hard negative samples
        rid_reps = [[rep] * self.topk for rep in rid_rep]
        rid_reps = list(chain(*rid_reps))
        rid_reps = torch.stack(rid_reps)    # [K*B, E]
        cid_reps = [[rep] * self.topk for rep in cid_rep]
        cid_reps = list(chain(*cid_reps))
        cid_reps = torch.stack(cid_reps)    # [K*B, E]
        loss += self.hard_criterion(cid_reps, hrid_rep, rid_reps)
        return loss, acc
