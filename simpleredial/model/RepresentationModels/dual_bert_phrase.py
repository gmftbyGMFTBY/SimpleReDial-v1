from model.utils import *

class BERTDualPhraseEncoder(nn.Module):

    '''Phrase-level extraction with GPT-2 LM Head as the query'''

    def __init__(self, **args):
        super(BERTDualPhraseEncoder, self).__init__()
        model = args['pretrained_model']
        gpt2_model = args['gpt2_lm_model']
        self.ctx_encoder = GPT2LMIRModel(model=gpt2_model)
        self.can_encoder = BertEmbedding(model=model)

        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_logits, cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        return cid_logits, cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

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

        batch_size = cid.shape[0]
        cid_logits, cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        # context lm loss
        shift_logits = cid_logtis[..., :-1, :],contiguous()
        shift_logits = cid[..., 1:].contiguous()
        lm_loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        # context lm acc
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum()
        token_acc = correct / num_targets

        # phrase-level extraction loss
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        return loss, acc
