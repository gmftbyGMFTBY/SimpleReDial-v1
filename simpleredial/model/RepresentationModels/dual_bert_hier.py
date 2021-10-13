from model.utils import *


class BERTDualHierarchicalEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualHierarchicalEncoder, self).__init__()
        model = args['pretrained_model']
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)

        self.fusion_head = nn.Sequential(
            nn.Linear(768*2, 768*2),
            nn.ReLU(),
            nn.Linear(768*2, 768)
        )

    def _encode(self, cids, rid, cids_mask, rid_mask, turn_length):
        cid_rep = self.ctx_encoder(cids, cids_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        cid_reps = torch.split(cid_rep, turn_length)
        return cid_reps, rid_rep

    def get_context_level_rep(self, cid_reps, turn_length):
        '''resort and generate the order, context length mask'''
        max_turn_length = max([len(i) for i in cid_reps])

        # padding by the turn_length
        reps, cid_mask = [], []    # [B, S]
        last_cid_rep = []
        for cid_rep in cid_reps:
            # cid_rep: L*[E]
            last_cid_rep.append(cid_rep[-1])
            # mask, [S], do not count the last utterance
            # m = torch.tensor([1] * (len(cid_rep) - 1) + [0] * (max_turn_length - len(cid_rep) + 1)).to(torch.bool)
            m = torch.tensor([1] * len(cid_rep) + [0] * (max_turn_length - len(cid_rep))).to(torch.bool)
            cid_mask.append(m)
            if len(cid_rep) < max_turn_length:
                zero_tensor = torch.zeros(1, 768).cuda()
                padding = [zero_tensor] * (max_turn_length - len(cid_rep))
                cid_rep = torch.cat([cid_rep] + padding)
            reps.append(cid_rep)
        reps = torch.stack(reps)    # [B, S, E]
        last_reps = torch.stack(last_cid_rep)    # [B, E]
        cid_mask = torch.stack(cid_mask)    # [B, S]

        # attention mechanism
        last_reps = last_reps.unsqueeze(1)    # [B, 1, E]
        attention_score = torch.bmm(last_reps, reps.permute(0, 2, 1)).squeeze(1)    # [B, S]
        attention_score /= np.sqrt(768)
        weight = torch.where(cid_mask != 0, torch.zeros_like(cid_mask), torch.ones_like(cid_mask)).cuda()    # [B, S]
        weight = weight * -1e3
        attention_score += weight
        attention_score = F.softmax(attention_score, dim=-1)

        attention_score = attention_score.unsqueeze(1)    # [B, 1, S]

        # generate the context level represetations
        # [B, 1, S] x [B, S, E]
        history_reps = torch.bmm(attention_score, reps).squeeze(1)    # [B, E]
        last_reps = last_reps.squeeze(1)
        reps = self.fusion_head(
            torch.cat([last_reps, history_reps], dim=-1)        
        )    # [B, E]
        return reps
    
    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        cid = cid.squeeze(0)    # [B, S]
        cid_mask = cid_mask.squeeze(0)

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze()
        return dot_product

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        turn_length = batch['turn_length']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_reps, rid_rep = self._encode(cid, rid, cid_mask, rid_mask, turn_length)
        cid_rep = self.get_context_level_rep(cid_reps, turn_length)

        dot_product = torch.matmul(cid_rep, rid_rep.t())    # [B, B]
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        acc_num = (dot_product.max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
