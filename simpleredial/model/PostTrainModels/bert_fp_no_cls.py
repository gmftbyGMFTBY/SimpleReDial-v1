from model.utils import *

class BERTFPNoCLSPostTrain(nn.Module):

    def __init__(self, **args):
        super(BERTFPNoCLSPostTrain, self).__init__()
        model = args['pretrained_model']
        self.model = BertForPreTraining.from_pretrained(model)
        self.criterion = nn.CrossEntropyLoss(ignore_index=-1)
        self.vocab_size = self.model.config.vocab_size

    def forward(self, batch):
        inpt = batch['ids']
        token_type_ids = batch['tids']
        attn_mask = batch['attn_mask']
        mask_labels = batch['mask_labels']

        # [B, S, V]; [B, E]
        output = self.model(
            input_ids=inpt,
            attention_mask=attn_mask,
            token_type_ids=token_type_ids,
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

        # placeholder
        cls_acc = 0.
        cls_loss = torch.tensor(0)
        return mlm_loss, cls_loss, token_acc, cls_acc
