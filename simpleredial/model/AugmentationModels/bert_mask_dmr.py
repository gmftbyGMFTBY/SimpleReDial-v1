from model.utils import *
from dataloader.util_func import *


class BERTMaskAugmentationDMRModel(nn.Module):

    '''DMR: dynamic mask ratio'''

    def __init__(self, **args):
        super(BERTMaskAugmentationDMRModel, self).__init__()
        self.args = args
        model = args['pretrained_model']
        self.model = BertForMaskedLM.from_pretrained(model)
        self.model.resize_token_embeddings(self.model.config.vocab_size+1)

        self.vocab = BertTokenizer.from_pretrained(model)
        self.special_tokens = self.vocab.convert_tokens_to_ids(['[PAD]', '[SEP]', '[CLS]'])
        self.unk, self.mask, self.pad = self.vocab.convert_tokens_to_ids(['[UNK]', '[MASK]', '[PAD]'])
        self.da_num = args['augmentation_t']
        self.ratio_list = np.arange(self.args['min_masked_lm_prob'], self.args['max_masked_lm_prob'], (self.args['max_masked_lm_prob']-self.args['min_masked_lm_prob'])/self.da_num)
        assert len(self.ratio_list) == self.da_num
    
    @torch.no_grad()
    def forward(self, batch):
        inpt = batch['ids']
        rest = []

        for i in range(self.da_num):
            ids = []
            for ii in deepcopy(inpt):
                if len(ii) == 2:
                    ii = [ii[0]] + [self.unk] * 2 + [ii[1]]
                mask_sentence_only_mask(ii, self.args['min_mask_num'], self.args['max_mask_num'], self.ratio_list[i], mask=self.mask, vocab_size=len(self.vocab), special_tokens=self.special_tokens)
                ii = torch.LongTensor(ii)
                ids.append(ii)
            ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
            mask = generate_mask(ids)
            ids, mask = to_cuda(ids, mask)

            logits = self.model(
                input_ids=ids,
                attention_mask=mask,
            )[0]    # [B, S, V]
            sent = self.generate_text(ids, F.softmax(logits, dim=-1))
            rest.append(sent)
        # rest: K*[B] -> B*[K]
        if batch['full'] is False:
            rest_ = []
            for i in range(len(batch['response'])):
                rest_.append([item[i] for item in rest])
        else:
            idx, length = 0, batch['length']
            rest_ = []
            for i in range(len(batch['context'])):
                l = length[i]
                pause = [item[idx:idx+l] for item in rest]
                pause_2 = []
                for jdx in range(len(pause[0])):
                    pause_2.append([item[jdx] for item in pause])
                rest_.append(pause_2)
                idx += l
        return rest_

    def generate_text(self, ids, logits):
        sentences = []
        for item, inpt in zip(logits, ids):
            inpt = inpt.tolist()
            tokens_ = torch.multinomial(item, num_samples=1).tolist()    # [S, K]
            tokens_ = [token[0] for token in tokens_]
            ts = [t if ot == self.mask else ot for t, ot in zip(tokens_, inpt) if ot not in self.special_tokens and t not in self.special_tokens]
            if self.args['lang'] == 'en':
                string = self.vocab.decode(ts)
            else:
                string = [self.vocab.convert_ids_to_tokens(t) for t in ts]
                string = ''.join(string).replace('##', '')
            sentences.append(string)
        return sentences
