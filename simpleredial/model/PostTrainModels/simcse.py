from model.utils import *

class SimCSE(nn.Module):

    '''two bert encoder are not shared, which is different from the original SimCSE model'''

    def __init__(self, **args):
        super(SimCSE, self).__init__()
        model = args['pretrained_model']
        self.temp = args['temp']
        self.encoder = BertEmbedding(model=model, add_tokens=1)
        self.args = args

    def _encode(self, ids, ids_mask):
        rep_1 = self.encoder(ids, ids_mask)
        rep_2 = self.encoder(ids, ids_mask)
        rep_1, rep_2 = F.normalize(rep_1), F.normalize(rep_2)
        return rep_1, rep_2

    @torch.no_grad()
    def get_embedding(self, ids, ids_mask):
        rep = self.encoder(ids, ids_mask)
        rep = F.normalize(rep)
        return rep

    def forward(self, batch):
        ids = batch['ids']
        ids_mask = batch['ids_mask']

        rep_1, rep_2 = self._encode(ids, ids_mask)
        # distributed samples collected
        rep_1s, rep_2s = distributed_collect(rep_1, rep_2)
        dot_product = torch.matmul(rep_1s, rep_2s.t())     # [B, B]
        dot_product /= self.temp
        batch_size = len(rep_1s)

        # constrastive loss
        mask = torch.zeros_like(dot_product)
        mask[range(batch_size), range(batch_size)] = 1. 
        loss_ = F.log_softmax(dot_product, dim=-1) * mask
        loss = (-loss_.sum(dim=1)).mean()
        
        # acc
        acc_num = (F.softmax(dot_product, dim=-1).max(dim=-1)[1] == torch.LongTensor(torch.arange(batch_size)).cuda()).sum().item()
        acc = acc_num / batch_size
        return loss, acc
