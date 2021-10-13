from model.utils import *
from .utils import *


'''GPT2 with gradient reverse layer for unlikelyhood training'''


class GPT2GRLModel(nn.Module):

    def __init__(self, **args):
        super(GPT2GRLModel, self).__init__()
        model = args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(model)
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.vocab = BertTokenizerFast.from_pretrained(model)
        self.pad, self.bos, self.eos = self.vocab.convert_tokens_to_ids(['[PAD]', '[CLS]', '[SEP]'])
        self.topk = args['topk']
        self.topp = args['topp']
        self.temp = args['temp']
        self.test_max_len = args['gen_max_len']
        self.test_min_len = args['gen_min_len']
        self.repetition_penalty = args['repetition_penalty']

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        gen_logits = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = gen_logits.logits
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl

    @torch.no_grad()
    def predict(self, batch):
        ids = batch['ids']
        ids_mask = batch['mask']
        logits = self.model.generate(
            input_ids=ids, 
            attention_mask=ids_mask,
            pad_token_id=self.pad,
            bos_token_id=self.bos,
            eos_token_id=self.eos,
            top_k=self.topk,
            top_p=self.topp,
            temperature=self.temp,
            forced_eos_token_id=True,
            do_sampling=True,
            max_length=self.test_max_len,
            min_length=self.test_min_len,
            repetition_penalty=self.repetition_penalty,
        )
        return logits

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['mask']

        batch_size = ids.shape[0]
        # [B, S, V]
        gen_logits = self.model(input_ids=ids, attention_mask=ids_mask)
        gen_logits = gen_logits.logits

        # generative loss
        # gen_logits: [B, S, V]; label: [B, S]
        shift_logits = gen_logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )

        # token acc
        chosen_tokens = torch.max(shift_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == shift_labels.view(-1)).to(torch.long)
        valid_mask = (shift_labels != 0).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()

        # negtive sample training
        neg_ids = batch['neg_ids']
        neg_ids_mask = batch['neg_ids_mask']
        neg_label = batch['neg_label']
        neg_gen_logits = self.model(
            input_ids=neg_ids, 
            attention_mask=neg_ids_mask
        )
        neg_gen_logits = neg_gen_logits.logits
        # reverse the gradient
        neg_gen_logits = GradientReverseFunction.apply(neg_gen_logits)
        shift_logits = neg_gen_logits[..., :-1, :].contiguous()
        shift_labels = neg_label[..., 1:].contiguous()
        loss += self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        return loss, gen_acc
