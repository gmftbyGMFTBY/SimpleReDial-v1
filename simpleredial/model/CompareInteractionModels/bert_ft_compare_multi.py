from model.utils import *
from dataloader.util_func import *

class BERTCompareMultiEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTCompareMultiEncoder, self).__init__()
        model = args['pretrained_model']
        self.model = BertSAModel.from_pretrained(model)
        self.cls = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, 2)
        )
        # add the [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        ids, sids, tids, lids, mask = batch['ids'], batch['sids'], batch['tids'], batch['lids'], batch['mask']
        
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            speaker_ids=sids,
        )[0]    # [B, S, E]
        logits = self.cls(output)    # [B, S, 2]
        loss = self.criterion(logits.view(-1, 2), lids.view(-1))

        # acc
        mask = lids != -100
        valid_num = mask.to(torch.float).sum().item()
        acc_num = ((logits.max(dim=-1)[1] == lids) & mask).sum().item()
        acc = acc_num / valid_num
        return loss, acc

    def predict(self, batch):
        ids  = batch['cids']
        sids = batch['sids']
        tids = batch['tids']
        mask = batch['mask']  
        lids = batch['lids']
        logits = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            speaker_ids=sids,
        )[0]
        logits = F.softmax(self.cls(logits), dim=-1)[0]    # [S, 2]
        # build gather index
        lids_index = (lids != -100).to(torch.float).nonzero()
        gather_index = [j.item() for _, j in lids_index]
        logits = logits[gather_index, :]    # [10, 2]
        assert len(logits) == 10
        logits = logits[:, 1]    # label 1 score
        return logits


class BERTCompareMultiCLSEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTCompareMultiCLSEncoder, self).__init__()
        model = args['pretrained_model']
        self.model = BertSAModel.from_pretrained(model)
        self.cls = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, 10)
        )
        # add the [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        ids, sids, tids, label, mask = batch['ids'], batch['sids'], batch['tids'], batch['label'], batch['mask']
        
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            speaker_ids=sids,
        )[0]    # [B, S, E]
        logits = self.cls(output[:, 0, :])    # [B, 10]
        loss = self.criterion(logits, label)

        # acc
        acc = (logits.max(dim=-1)[1] == label).to(torch.float).mean().item()
        return loss, acc

    def predict(self, batch):
        ids  = batch['cids']
        sids = batch['sids']
        tids = batch['tids']
        mask = batch['mask']  
        logits = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            speaker_ids=sids,
        )[0]
        logits = F.softmax(self.cls(logits[:, 0, :]), dim=-1)[0]
        return logits


class BERTCompareMultiENSEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTCompareMultiENSEncoder, self).__init__()
        model = args['pretrained_model']
        self.model = BertSAModel.from_pretrained(model)
        self.cls = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, 2)
        )
        self.cls2 = nn.Sequential(
            nn.Dropout(p=args['dropout']) ,
            nn.Linear(768, 10)
        )
        # add the [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)
        self.criterion = nn.CrossEntropyLoss()

    def forward(self, batch):
        ids, sids, tids, lids, mask, label = batch['ids'], batch['sids'], batch['tids'], batch['lids'], batch['mask'], batch['label']
        
        output = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            speaker_ids=sids,
        )[0]    # [B, S, E]

        # loss 1
        logits2 = self.cls2(output[:, 0, :])    # [B, 10]
        loss = self.criterion(logits2, label)
        cls_acc = (logits2.max(dim=-1)[1] == label).to(torch.float).mean().item()

        # loss 2
        logits = self.cls(output)    # [B, S, 2]
        loss += self.criterion(logits.view(-1, 2), lids.view(-1))
        return loss, cls_acc

    def predict(self, batch):
        ids  = batch['cids']
        sids = batch['sids']
        tids = batch['tids']
        mask = batch['mask']  
        logits = self.model(
            input_ids=ids,
            attention_mask=mask,
            token_type_ids=tids,
            speaker_ids=sids,
        )[0]
        logits = F.softmax(self.cls(logits[:, 0, :]), dim=-1)[0]
        return logits
