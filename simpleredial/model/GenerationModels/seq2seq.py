from model.utils import *
from .utils import *


class BERTSeq2SeqEncoder(nn.Module):

    def __init__(self, vocab, **args):
        super(BERTSeq2SeqEncoder, self).__init__()
        model = args['pretrained_model']
        self.vocab = vocab

        self.model = Seq2SeqModel(model, self.vocab.cls_token_id)
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=0)

    def _encode(self, cid, rid, cid_mask, rid_mask):
        gen_logits, _ = self.model(cid, cid_mask, rid, rid_mask)
        return gen_logits

    @torch.no_grad()
    def predict(self, batch):
        # generate
        cid = batch['ids'].unsqueeze(0)    # [1, S]
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        output = self.model.predict(cid)
        return output

    def forward(self, batch):
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        labels = batch['rids']

        batch_size = cid.shape[0]
        gen_logits = self._encode(cid, rid, cid_mask, rid_mask)

        # generative loss
        # gen_logits: [B, S, V]; label: [B, S]
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = labels[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        # token acc
        chosen_tokens = torch.max(shift_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == shift_labels.view(-1)).to(torch.float)
        counter, sum_ = 0, 0
        for i, j in zip(shift_labels.view(-1), gen_acc):
            if i != 0:
                sum_ += 1
                counter += j.item()
        gen_acc = counter/sum_
        return loss, gen_acc
