from model.utils import *

'''Speraker-aware Bert Cross-encoder'''

class SABERTRetrieval(nn.Module):

    def __init__(self, **args):
        super(SABERTRetrieval, self).__init__()
        model = args['pretrained_model']
        self.model = BertSAModel.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.cls = nn.Linear(768, 1)

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        speaker_type_ids = batch['sids']
        attn_mask = batch['mask']

        logits = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
            speaker_ids=speaker_type_ids,
        )[0]    # [B, S, E]
        logits = self.cls(logits[:, 0, :]).squeeze(dim=-1)    # [B]
        return logits
