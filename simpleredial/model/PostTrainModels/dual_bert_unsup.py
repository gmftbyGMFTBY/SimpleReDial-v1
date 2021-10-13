from model.utils import *

class BERTDualUnsupervisedEncoder(nn.Module):
    
    '''Unsupervised to warmup the ctx and res encoders'''

    def __init__(self, **args):
        super(BERTDualUnsupervisedEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_rep, rid_rep

    def calculate_one_batch_clloss(self, cid_rep, rid_rep):
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B] or [B, 2*B]
        dot_product /= self.temp
        batch_size = len(cid_rep)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        return loss, dot_product
    
    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']
        batch_size = len(ids)

        cid_rep, rid_rep = self._encode(ids, ids, ids_mask, ids_mask)
        cid_rep_1, rid_rep_1 = self._encode(ids, ids, ids_mask, ids_mask)
        loss = 0
        loss_, dot_product_1 = self.calculate_one_batch_clloss(
            cid_rep, 
            torch.cat([cid_rep_1, rid_rep, rid_rep_1], dim=0),    # [B, 3*E]
        )
        loss += loss_
        loss_, dot_product_2 = self.calculate_one_batch_clloss(
            rid_rep, 
            torch.cat([rid_rep_1, cid_rep, cid_rep_1], dim=0),    # [B, 3*E]
        )
        loss += loss_

        # acc
        acc_num = (F.softmax(dot_product_1, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc_num += (F.softmax(dot_product_2, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size / 2
        return torch.tensor(0.0), loss, 0., acc
