from model.utils import *

class BERTDualAdvEncoder(nn.Module):

    def __init__(self, **args):
        super(BERTDualAdvEncoder, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.ctx_encoder = BertEmbedding(model=model, add_tokens=1)
        self.can_encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

        self.dc_layer = nn.Sequential(
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Dropout(p=args['dropout']),
            nn.Linear(768, 768),
            nn.ReLU(),
            nn.Linear(768, 1),
        )
        self.dc_criterion = nn.BCEWithLogitsLoss()

    def _encode(self, cid, rid, cid_mask, rid_mask):
        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        # cosine similarity
        cid_rep, rid_rep = F.normalize(cid_rep), F.normalize(rid_rep)
        return cid_rep, rid_rep

    @torch.no_grad()
    def get_cand(self, ids, attn_mask):
        rid_rep = self.can_encoder(ids, attn_mask)
        return rid_rep

    @torch.no_grad()
    def get_ctx(self, ids, attn_mask):
        cid_rep = self.ctx_encoder(ids, attn_mask)
        return cid_rep

    @torch.no_grad()
    def predict(self, batch):
        cid = batch['ids']
        cid_mask = torch.ones_like(cid)
        rid = batch['rids']
        rid_mask = batch['rids_mask']

        batch_size = rid.shape[0]
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)
        dot_product = torch.matmul(cid_rep, rid_rep.t()).squeeze(0)
        return dot_product
    
    def forward(self, batch):
        '''context domain - label 0; response domain - label 1'''
        cid = batch['ids']
        rid = batch['rids']
        cid_mask = batch['ids_mask']
        rid_mask = batch['rids_mask']
        l = batch['l']
        batch_size = len(cid)

        # encode
        cid_rep, rid_rep = self._encode(cid, rid, cid_mask, rid_mask)

        # constrastive loss
        dot_product = torch.matmul(cid_rep, rid_rep.t()) 
        dot_product /= self.temp

        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()

        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        
        # domain classifier
        ## flip gradient, l is the progress of the training
        cid_rep = GradientReverseFunction.apply(cid_rep, l)
        rid_rep = GradientReverseFunction.apply(rid_rep, l)

        dc_rep = torch.cat([cid_rep, rid_rep], dim=0)  # [2*B, E]
        dc_label = [0.] * batch_size + [1.] * batch_size
        ## shuffle
        random_idx = list(range(batch_size*2))
        random.shuffle(random_idx)
        dc_label = torch.tensor([dc_label[i] for i in random_idx]).cuda()
        dc_rep = torch.stack([dc_rep[i] for i in random_idx])
        dc_rep = self.dc_layer(dc_rep).squeeze(dim=-1)    # [2*B]
        
        # loss for domain classifier (dc)
        dc_loss = self.dc_criterion(dc_rep, dc_label) 
        dc_acc = ((torch.sigmoid(dc_rep) > 0.5) == dc_label).float().mean().item()
        return loss, dc_loss, acc, dc_acc
