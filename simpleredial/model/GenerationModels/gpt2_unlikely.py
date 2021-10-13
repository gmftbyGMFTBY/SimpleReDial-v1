from model.utils import *
from .utils import *


class GPT2UnlikelyModel(nn.Module):

    def __init__(self, **args):
        super(GPT2UnlikelyModel, self).__init__()
        gpt2_model = args['pretrained_model']
        bert_model = args['bert_pretrained_model']
        self.gpt2_model = GPT2LMHeadModel.from_pretrained(gpt2_model)
        self.bert_model = BertGenerationOnGPT2Decoder.from_pretrained(bert_model)
        # pad token is 0
        self.gen_loss_fct = nn.CrossEntropyLoss(ignore_index=0)
        self.vocab = BertTokenizerFast.from_pretrained(gpt2_model)
        self.pad, self.bos, self.eos = self.vocab.convert_tokens_to_ids(['[PAD]', '[CLS]', '[SEP]'])
        self.topk = args['topk']
        self.topp = args['topp']
        self.temp = args['temp']
        self.test_max_len = args['gen_max_len']
        self.test_min_len = args['gen_min_len']
        self.repetition_penalty = args['repetition_penalty']
        self.alpha = args['alpha']

    @torch.no_grad()
    def calculate_ppl(self, ids, ids_mask, label):
        gpt2_output = self.gpt2_model(
            input_ids=ids, 
            attention_mask=ids_mask,
            output_hidden_states=True    
        )
        gpt2_hidden_states = gpt2_output.hidden_states[-1]    # [B, S, E]
        gpt2_hidden_states = gpt2_hidden_states.mean(dim=1)    # [B, E]
        bert_output = self.bert_model(
            input_ids=ids, 
            attention_mask=ids_mask,
            gpt2_hidden_states=gpt2_hidden_states,
        )
        bert_logits = bert_output.logits    # [B, S, V]
        shift_logits = bert_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        ppl = math.exp(loss.item())
        return ppl

    def calculate_token_acc(self, shift_logits, shift_labels): 
        # token acc
        chosen_tokens = torch.max(shift_logits, dim=-1)[1]    # [B, S-1]
        gen_acc = (chosen_tokens.view(-1) == shift_labels.view(-1)).to(torch.long)
        valid_mask = (shift_labels != 0).view(-1)
        valid_tokens = gen_acc & valid_mask
        gen_acc = valid_tokens.sum().item() / valid_mask.sum().item()
        return gen_acc

    def get_lm_loss(self, shift_logits, shift_labels):
        loss = self.gen_loss_fct(
            shift_logits.view(-1, shift_logits.size(-1)), 
            shift_labels.view(-1)
        )
        return loss

    @torch.no_grad()
    def predict(self, batch):
        ids = batch['ids']
        ids_mask = batch['mask']
        # gpt2 inference
        gpt2_output = self.gpt2_model(
            input_ids=ids, 
            attention_mask=ids_mask,
            output_hidden_states=True    
        )
        gpt2_hidden_states = gpt2_output.hidden_states[-1]    # [B, S, E]
        gpt2_hidden_states = gpt2_hidden_states.mean(dim=1)    # [B, E]
        # bert inference
        model_kwargs = {'gpt2_hidden_states': gpt2_hidden_states}
        logits = self.bert_model.generate(
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
            **model_kwargs,
        )
        return logits

    def gpt2_forward(self, ids, ids_mask):
        gpt2_output = self.gpt2_model(
            input_ids=ids, 
            attention_mask=ids_mask,
            output_hidden_states=True    
        )
        gpt2_logits = gpt2_output.logits    # [B, S, V]
        gpt2_hidden_states = gpt2_output.hidden_states[-1]    # [B, S, E]
        
        shift_logits = gpt2_logits[..., :-1, :].contiguous()
        shift_labels = ids[..., 1:].contiguous()

        gpt2_gen_acc = self.calculate_token_acc(
            shift_logits, 
            shift_labels
        )
        gpt2_loss = self.get_lm_loss(
            shift_logits, 
            shift_labels
        )
        # [B, E], scalar, tensor scalar
        gpt2_hidden_states = gpt2_hidden_states.mean(dim=1)    # [B, E]
        return gpt2_hidden_states, gpt2_gen_acc, gpt2_loss

    def bert_forward(self, gpt2_hidden_states, ids, ids_mask, label):
        bert_output = self.bert_model(
            input_ids=ids, 
            attention_mask=ids_mask,
            gpt2_hidden_states=gpt2_hidden_states,
        )
        bert_logits = bert_output.logits    # [B, S, V]
        shift_logits = bert_logits[..., :-1, :].contiguous()
        shift_labels = label[..., 1:].contiguous()

        bert_gen_acc = self.calculate_token_acc(
            shift_logits, 
            shift_labels
        )
        bert_loss = self.get_lm_loss(
            shift_logits, 
            shift_labels
        )
        return bert_gen_acc, bert_loss

    def forward(self, batch):
        # ===== input ===== #
        gpt2_ids       = batch['gpt2_ids']
        gpt2_mask      = batch['gpt2_mask']
        bert_label     = batch['bert_label']
        
        neg_gpt2_ids   = batch['neg_gpt2_ids']
        neg_gpt2_mask  = batch['neg_gpt2_mask']
        neg_bert_label = batch['neg_bert_label']
        # ===== input ===== # 

        loss = 0
        # positive samples feedforward
        gpt2_hidden_states, gpt2_token_acc, gpt2_loss = self.gpt2_forward(gpt2_ids, gpt2_mask)
        bert_token_acc, bert_loss = self.bert_forward(gpt2_hidden_states, gpt2_ids, gpt2_mask, bert_label)
        loss += gpt2_loss + bert_loss

        # negative samples feedforward
        gpt2_hidden_states, _, _ = self.gpt2_forward(neg_gpt2_ids, neg_gpt2_mask)
        # reverse the gradient for reversed loss optimization
        gpt2_hidden_states = GradientReverseFunction.apply(gpt2_hidden_states, 1.)
        _, neg_bert_loss = self.bert_forward(gpt2_hidden_states, neg_gpt2_ids, neg_gpt2_mask, neg_bert_label)
        loss += self.alpha * neg_bert_loss
        # bert_loss for calculating the ppl
        return loss, bert_token_acc, bert_loss
