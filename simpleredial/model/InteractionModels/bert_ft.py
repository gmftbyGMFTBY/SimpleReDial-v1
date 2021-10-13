from model.utils import *

class BERTRetrieval(nn.Module):

    def __init__(self, **args):
        super(BERTRetrieval, self).__init__()
        model = args['pretrained_model']
        # bert-fp pre-trained model need to resize the token embedding
        self.model = BertForSequenceClassification.from_pretrained(model, num_labels=1)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        attn_mask = batch['mask']

        logits = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
        )[0]    # [B, 1]
        return logits.squeeze(dim=-1)
