from model.utils import *


'''seq2seq (bert2bert)'''


class Seq2SeqModel(nn.Module):

    def __init__(self, model, cls):
        super(Seq2SeqModel, self).__init__()
        self.cls_token_id = cls
        self.model = EncoderDecoderModel.from_encoder_decoder_pretrained(model, model)
        # bert-ft pretrained checkpoint
        self.model.encoder.resize_token_embeddings(self.model.encoder.config.vocab_size+1)
        self.model.decoder.resize_token_embeddings(self.model.decoder.config.vocab_size+1)

    def forward(self, cid, cid_mask, rid, rid_mask):
        outputs = self.model(
            input_ids=cid, 
            attention_mask=cid_mask, 
            decoder_input_ids=rid, 
            decoder_attention_mask=rid_mask
        )
        logits = outputs.logits    # [B, S, V]
        hidden = outputs.encoder_last_hidden_state.mean(dim=1)
        return logits, hidden

    def predict(self, cid):
        # Greedy Decoding
        output = self.model.generate(
            cid, 
            do_sample=True,
            decoder_start_token_id=self.cls_token_id,
        )
        return output[0]

    def load_bert_model(self, state_dict):
        # decoder has the different parameters
        new_state_dict = OrderedDict()
        for k, v in state_dict.items():
            new_state_dict[k] = v
        new_state_dict['embeddings.position_ids'] = torch.arange(512).expand((1, -1))
        self.model.encoder.load_state_dict(new_state_dict)
