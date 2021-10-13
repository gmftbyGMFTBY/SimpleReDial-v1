from header import *
from .utils import *
from .util_func import *


class EvaluationDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.vocab = vocab
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        self.cls = self.vocab.convert_tokens_to_ids('[CLS]')
        self.unk = self.vocab.convert_tokens_to_ids('[UNK]')
        self.mask = self.vocab.convert_tokens_to_ids('[MASK]')

        self.special_tokens = set([self.pad, self.sep, self.cls, self.unk, self.mask])
        self.max_ctx_length = self.args['max_ctx_length']

        data, self.responses = read_essay_dataset(path)
        self.data = []
        last_e_idx, last_e_id = -1, -1
        counter = 0
        for _, p_id, _, sentence in tqdm(data):
            ctx = [s for _, _, _, s in data[last_p_idx:counter][-self.max_ctx_length:]]
            pos_res = data[counter][-1]
            self.data.append({
                'ctx': ctx,
                'res': pos_res,
            })
            if p_id != last_p_id:
                last_p_id = p_id
                last_p_idx = counter
            counter += 1

    def __len__(self):
        return len(self.table)

    def __getitem__(self, i):
        bundle = self.data[i]
        ctx, pos_res = bundle['ctx'], bundle['res']
        neg_res = random.choice(self.responses)

        ids = self.vocab.encode_plus(ctx + [pos_res, neg_res], add_special_tokens=False)['input_ids']
        ctx_ids = ids[:-2]
        pos_res_ids = ids[-2]
        neg_pos_ids = ids[-1]
        ids_ = [self.cls]
        for u in ctx_ids:
            ids_.extend(u + [self.sep])
        ids_.pop()
        ids_ = [self.cls] + ids_ + [self.sep] + ids + [self.sep]

        ids, tids, label, pert_label = [], [], [], []
        ids_, tids_, label_, pert_label_ = self.pertubation(ctx_ids, pos_res_ids)
        ids.append(ids_)
        tids.append(tids_)
        label.append(label_)
        pert_label.append(pert_label_)
        ids_, tids_, label_, pert_label = self.pertubation(ctx_ids, neg_res_ids)
        ids.append(ids_)
        tids.append(tids_)
        label.append(label_)
        pert_label.append(pert_label_)
        return ids, tids, label, pert_label, [1, 0]

    def pertubation(self, ctx_ids, ids)
        # positive
        ratio = random.random()
        if ratio < 0.333:
            # delete
            ids, label = delete(
                ids, 
                delete_ratio=self.args['delete_ratio'],
                min_delete_num=self.args['min_delete_num'],
                special_tokens=[self.cls, self.sep, self.unk, self.mask],
            )
        elif 0.3333 < ratio < 0.6667:
            # duplicate
            ids, label = duplicate(
                ids,
                duplicate_ratio=self.args['duplicate_ratio'],
                min_duplicate_num=self.args['min_duplicate_num'],
                special_tokens=[self.cls, self.sep, self.unk, self.mask],
            )
        else:
            # replace
            ids, label = replacement(
                ids,
                replace_ratio=self.args['replace_ratio'],
                min_replace_num=self.args['min_replace_num'],
                vocab_size=len(self.vocab),
                special_tokens=[self.cls, self.sep, self.unk, self.mask],
            )
        tids_ = [0] * (len(ids_) + 2) + [1] * (len(pos_res_ids) + 1)
        label = [-1] + label + [-1] + [-1] * (len(ids) + 1)
        return ids_, tids_, label

    def save(self):
        data = torch.save((self.data, self.table), self.pp_path)
        print(f'[!] save preprocessed dataset into {self.pp_path}; size: {len(self.table)}')
        
    def collate(self, batch):
        ids, tids, labels, pert_label, cls_labels = [], [], [], []
        for ids_, tids_, labels_, cls_label_ in batch:
            ids.extend(ids_)
            tids.extend(tids_)
            labels.extend(labels_)
            cls_labels.extend(cls_label_)
        ids = [torch.LongTensor(i) for i in ids]
        tids = [torch.LongTensor(i) for i in tids]
        labels = [torch.LongTensor(i) for i in labels]
        cls_label = torch.LongTensor(cls_labels)

        ids = pad_sequence(ids, batch_first=True, padding_value=self.pad)
        tids = pad_sequence(tids, batch_first=True, padding_value=self.pad)
        labels = pad_sequence(labels, batch_first=True, padding_value=-1)    # pad is not calculated for MLM
        attn_mask = generate_mask(ids)
        ids, tids, labels, attn_mask, cls_labels = to_cuda(ids, tids, labels, attn_mask, cls_labels)
        return {
            'ids': ids, 
            'tids': tids, 
            'labels': labels, 
            'attn_mask': attn_mask, 
            'cls_label': cls_labels,
        }
