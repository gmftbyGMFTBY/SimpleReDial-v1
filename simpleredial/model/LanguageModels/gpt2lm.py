from model.utils import *
from dataloader.util_func import *

class GPT2LM(nn.Module):

    def __init__(self, **args):
        super(GPT2LM, self).__init__()
        tokenizer, pretrained_model = args['tokenizer'], args['pretrained_model']
        self.model = GPT2LMHeadModel.from_pretrained(tokenizer)
        self.vocab = BertTokenizer.from_pretrained(pretrained_model)
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.args = args

    def _gpt2_convert_to_ids(self, context, sentence):
        cids, rids = self.vocab.batch_encode_plus([context, sentence], add_special_tokens=False)['input_ids']
        truncate_pair(cids, rids, self.args['max_len']+1)
        ids = [self.cls] + cids + rids + [self.sep]
        label = [-100] * (len(cids) + 1) + rids + [-100]
        ids = torch.LongTensor(ids)
        label = torch.LongTensor(label)
        ids, label = to_cuda(ids, label)
        return ids, label 

    @torch.no_grad()
    def predict(self, batch):
        candidates = batch['candidates']
        context = batch['context']
        rest = []
        for sentence in candidates:
            ids, label = self._gpt2_convert_to_ids(context, sentence)
            loss = self.model(ids, labels=label)[0]    # loss
            ppl = math.exp(loss.item())
            rest.append(ppl)
        # normalization
        norm = torch.tensor(rest)
        norm = 1 - (norm - norm.min()) / (norm.max() - norm.min())
        rest = [(ppl, norm_) for ppl, norm_ in zip(rest, norm.tolist())]
        return rest
