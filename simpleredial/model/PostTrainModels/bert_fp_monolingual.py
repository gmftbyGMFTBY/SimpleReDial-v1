from model.utils import *

class BERTMonolingualPostTrain(nn.Module):

    '''use the weight of the bert-fp, and only the masked lm loss is used'''

    def __init__(self, **args):
        super(BERTMonolingualPostTrain, self).__init__()
        model = args['pretrained_model']
        p = args['dropout']

        self.model = BertForPreTraining.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)    # [EOS]
        self.model.cls.seq_relationship = nn.Sequential(
            nn.Dropout(p=p),
            nn.Linear(768, 3)
        )
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, batch):
        inpt = batch['ids']
        attn_mask = batch['attn_mask']
        mask_labels = batch['mask_labels']

        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
        )
        prediction_scores = output.prediction_logits
        mlm_loss = self.criterion(
            prediction_scores.view(-1, self.vocab_size),
            mask_labels.view(-1),
        ) 

        # calculate the acc
        not_ignore = mask_labels.ne(-1)
        num_targets = not_ignore.sum().item()
        correct = (prediction_scores.max(dim=-1)[1] == mask_labels) & not_ignore
        correct = correct.sum().item()
        token_acc = correct / num_targets
        return mlm_loss, torch.tensor(0.0), token_acc, 0.
