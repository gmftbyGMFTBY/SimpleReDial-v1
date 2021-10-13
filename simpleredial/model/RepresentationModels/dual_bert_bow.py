from model.utils import *

class BERTDualBOWEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualBOWEncoder, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']
        self.bow_loss_lambda = args['bow_loss_lambda']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.ctx_vocab_size = self.ctx_encoder.model.config.vocab_size
        self.ctx_lm_head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, self.ctx_vocab_size),
        )
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_vocab_size = self.can_encoder.model.config.vocab_size
        self.can_lm_head = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, self.can_vocab_size),
        )
        self.vocab = BertTokenizer.from_pretrained(model)
        self.vocab.add_tokens(['[EOS]'])
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.eos = self.vocab.convert_tokens_to_ids('[EOS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sp_tokens = [self.cls, self.sep, self.eos, self.unk, self.pad]

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        cid_logits = F.log_softmax(self.ctx_lm_head(cid_rep), dim=-1)
        rid_rep = self.can_encoder(rid, rid_mask)
        rid_logits = F.log_softmax(self.can_lm_head(cid_rep), dim=-1)
        return cid_rep, rid_rep, cid_logits, rid_logits

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
        cid_rep, rid_rep, _, _ = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product

    def convert_to_bow_label(self, ids):
        # remove the duplicate, and the special tokens; ids: [B, S]
        labels = []
        for item in ids:
            item = list(set(item.tolist()) - set(self.sp_tokens))
            labels.append(item)
        return labels
    
    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']

        batch_size = cid.shape[0]
        cid_rep, rid_rep, cid_logits, rid_logits = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t())     # [B, B]

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # bow loss
        # *_logits: [B, V]
        # context use the response as the label
        ctx_label = self.convert_to_bow_label(rid)
        res_label = self.convert_to_bow_label(cid)
        bow_loss = 0
        for c_log, r_log, c_label, r_label in zip(cid_logits, rid_logits, ctx_label, res_label):
            # c_log/r_log: [V]; c_label/r_label: a list of tokens that should be predicted
            bow_loss -= c_log[r_label].sum()
            bow_loss -= r_log[c_label].sum()
        bow_loss /= batch_size * 2
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size

        loss += self.bow_loss_lambda * bow_loss

        return loss, acc
