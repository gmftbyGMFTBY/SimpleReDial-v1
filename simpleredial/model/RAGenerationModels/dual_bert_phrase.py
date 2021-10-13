from model.utils import *

class BERTDualPhraseEncoder(nn.Module):

    '''Phrase-level extraction with GPT-2 LM Head as the query'''

    def __init__(self, **args):
        super(BERTDualPhraseEncoder, self).__init__()
        model = args['pretrained_model']
        gpt2_model = args['gpt2_lm_model']
        self.vocab = BertTokenizer.from_pretrained(model)
        self.pad = self.vocab.pad_token_id
        self.ctx_encoder = GPT2LMIRModel(model=gpt2_model)
        self.can_encoder = BertFullEmbedding(model=model)
        self.criterion = nn.CrossEntropyLoss(ignore_index=self.pad)

        p = args['dropout']
        self.proj_query = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 768),
        )

    def _encode(self, cid, cid_mask):
        cid_logits, cid_rep = self.ctx_encoder(cid, cid_mask)
        cid_rep = self.proj_query(cid_proj)
        rid_rep = self.can_encoder(cid, cid_mask)
        return cid_logits, cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)    # [B, S, E]
        embeddings, texts = [], []
        for rid_rep_, ids_, mask in zip(rid_rep, ids, attn_mask):
            # rid_rep_: [S, E]; ids_: [S]; mask: [S]
            num_nonzero = ids_.nonzero().size()[0]
            # ignore [CLS] and [SEP]
            for i in range(1, num_nonzero):
                embeddings.append(rid_rep_[i])
                text = ''.join(self.vocab.decode(ids_[i:num_nonzero-1]).split())
                texts.append(text)
        embeddings = torch.stack(embeddings)    # [N, E]
        return embeddings, texts 

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        batch_size = cid.shape[0]
        _, cid_rep, rid_rep = self._encode(cid, cid_mask)
        # phrase-level extraction loss
        rid_rep = rid_rep[:, 1:, :]
        cid_rep = cid_rep[:, :-1, :]
        # cid_rep: [B, S-1, E]; rid_rep: [B, S-1, E]
        dot_product = torch.bmm(cid_rep, rid_rep.permute(0, 2, 1)).squeeze(0)    # [S-1, S-1]
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(len(dot_product))).cuda()).sum().item()
        acc = acc_num / len(dot_product)
        return acc
    
    def forward(self, batch):
        cid = batch['ids']
        cid_mask = batch['ids_mask']    # [B, S]

        batch_size = cid.shape[0]
        cid_logits, cid_rep, rid_rep = self._encode(cid, cid_mask)
        # context lm loss
        shift_logits = cid_logits[..., :-1, :].contiguous()
        shift_labels = cid[..., 1:].contiguous()
        lm_loss = self.criterion(
            shift_logits.view(-1, shift_logits.size(-1)),
            shift_labels.view(-1)
        )
        # context lm acc
        _, preds = shift_logits.max(dim=-1)
        not_ignore = shift_labels.ne(self.pad)
        num_targets = not_ignore.long().sum().item()
        correct = (shift_labels == preds) & not_ignore
        correct = correct.float().sum().item()
        token_acc = correct / num_targets

        # phrase-level extraction loss
        rid_rep = rid_rep[:, 1:, :]   # [B, S-1, E]
        cid_mask = cid_mask[:, 1:]    # [B, S-1]
        cid_rep = cid_rep[:, :-1, :]  # [B, S-1, E]
        # cid_rep: [B, S-1, E]; rid_rep: [B, S-1, E]
        dot_product = torch.bmm(cid_rep, rid_rep.permute(0, 2, 1))    # [B, S-1, S-1]
        mask = torch.zeros_like(dot_product)    # [B, S-1, S-1]
        mask[:, range(batch_size), range(batch_size)] = 1. 
        # ignore the pad
        length = len(cid_mask[0])
        for i in range(len(mask)):
            num_nonzero = cid_mask[i].nonzero().size(0)
            # -1 means ignore the last [SEP] token, which is useless
            mask[i][range(num_nonzero-1, length), range(num_nonzero-1, length)] = 0.
        loss_ = F.log_softmax(dot_product, dim=-1) * mask    # [B, S-1, S-1]
        loss = -loss_.sum(dim=2)    # [B, S-1]
        loss = loss.view(-1).mean()    # [B*(S-1)]
        
        # acc
        acc_num, s = 0, 0
        for dp, mask_, cid_mask_ in zip(dot_product, mask, cid_mask):
            # dp: [S-1, S-1]; mask_: [S-1, S-1]
            num_nonzero = cid_mask[i].nonzero().size(0)
            dp = dp[:num_nonzero, :num_nonzero]
            acc_num += (F.softmax(dp, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(num_nonzero)).cuda()).sum().item()
            s += num_nonzero
        acc = acc_num / s
        return lm_loss, loss, token_acc, acc
