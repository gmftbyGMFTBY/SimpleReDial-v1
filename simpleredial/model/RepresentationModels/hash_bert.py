from model.utils import *

class HashBERTDualEncoder(nn.Module):
    
    def __init__(self, **args):
        super(HashBERTDualEncoder, self).__init__()
        self.args = args
        self.hash_code_size = args['hash_code_size']
        self.hidden_size = args['hidden_size']
        model = args['pretrained_model']
        self.gray_num = args['gray_cand_num']
        dropout = args['dropout']
        self.hash_loss_scale = args['hash_loss_scale']
        self.kl_loss_scale = args['kl_loss_scale']
        self.vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
        self.pad = self.vocab.convert_tokens_to_ids('[PAD]')
        self.sep = self.vocab.convert_tokens_to_ids('[SEP]')
        
        # dual bert pre-trained model
        self.ctx_encoder = BertEmbedding(model=model)
        self.can_encoder = BertEmbedding(model=model)
        
        gpu_ids = str(args['trainable_bert_layers'])
        self.trainable_layers = [f'encoder.layer.{i}' for i in gpu_ids.split(',')]
        inpt_size = self.ctx_encoder.model.config.hidden_size
        self.ctx_hash_encoder = nn.Sequential(
            nn.Linear(inpt_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, self.hash_code_size),
        )
        self.can_hash_encoder = nn.Sequential(
            nn.Linear(inpt_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, self.hash_code_size),
        )
        self.ctx_hash_decoder = nn.Sequential(
            nn.Linear(self.hash_code_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, inpt_size),
        )
        self.can_hash_decoder = nn.Sequential(
            nn.Linear(self.hash_code_size, self.hidden_size),
            nn.LeakyReLU(),
            nn.Dropout(p=dropout),
            nn.Linear(self.hidden_size, inpt_size),
        )

    def freeze_some_layers(self):
        if self.args['trainable_bert_layers'] == 'all':
            print(f'[!] DONOT Freeze the parameters of the bert')
            return
        # freeze context encoder parameters
        for key, param in self.ctx_encoder.named_parameters():
            for trainable_name in self.trainable_layers:
                if trainable_name in key:
                    break
            else:
                param.requires_grad_(False)
        # freeze candidate encoder parameters
        for key, param in self.can_encoder.named_parameters():
            for trainable_name in self.trainable_layers:
                if trainable_name in key:
                    break
            else:
                param.requires_grad_(False)
        print(f'[!] freeze the paremeters in some layers')

    def compact_binary_vectors(self, ids):
        # ids: [B, D]
        ids = ids.cpu().numpy().astype('int')
        ids = np.split(ids, int(ids.shape[-1]/8), axis=-1)
        ids = np.ascontiguousarray(
            np.stack(
                [np.packbits(i) for i in ids]    
            ).transpose().astype('uint8')
        )
        return ids

    @torch.no_grad()
    def get_cand(self, ids, ids_mask):
        self.can_encoder.eval()
        rid_rep = self.can_encoder(ids, ids_mask)
        hash_code = torch.sign(self.can_hash_encoder(rid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        hash_code = self.compact_binary_vectors(hash_code)
        hash_code = torch.from_numpy(hash_code)
        return hash_code
    
    @torch.no_grad()
    def get_ctx(self, ids, ids_mask):
        self.ctx_encoder.eval()
        cid_rep = self.ctx_encoder(ids, ids_mask)
        hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))
        hash_code = torch.where(
            hash_code == -1, 
            torch.zeros_like(hash_code), 
            hash_code,
        )
        hash_code = self.compact_binary_vectors(hash_code)
        return hash_code

    @torch.no_grad()
    def predict(self, batch):
        if 'context' in batch and 'responses' in batch:
            context = batch['context']
            responses = batch['responses']
            cid, cid_mask = self.totensor([context], ctx=True)
            rid, rid_mask = self.totensor(responses, ctx=False)
        elif 'ids' in batch and 'rids' in batch:
            cid, rid = batch['ids'], batch['rids']
            cid = cid.unsqueeze(0)
            cid_mask = None
            rid_mask = batch['rids_mask']
        else:
            raise Exception(f'[!] Unknow batch data')

        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)

        # [768]; [B, 768] -> [H]; [B, H]
        ctx_hash_code = torch.sign(self.ctx_hash_encoder(cid_rep))    # [1, Hash]
        can_hash_code = torch.sign(self.can_hash_encoder(rid_rep))    # [B, Hash]
        matrix = torch.matmul(ctx_hash_code, can_hash_code.t()).squeeze(0)    # [B]
        # minimal distance -> better performance 
        distance = -0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); distance: [B]
        return distance
        
    def forward(self, batch):
        if 'context' in batch and 'responses' in batch:
            context = batch['context']
            responses = batch['responses']
            cid, cid_mask = self.totensor(context, ctx=True)
            rid, rid_mask = self.totensor(responses, ctx=False)
        elif 'ids' in batch and 'rids' in batch:
            cid = batch['ids']
            cid_mask = batch['ids_mask']
            rid = batch['rids']
            rid_mask = batch['rids_mask']
        else:
            raise Exception(f'[!] Unknow batch data')

        cid_rep = self.ctx_encoder(cid, cid_mask)
        rid_rep = self.can_encoder(rid, rid_mask)
        
        batch_size = cid_rep.shape[0]

        ctx_hash_code = self.ctx_hash_encoder(cid_rep)    # [B, H]
        can_hash_code = self.can_hash_encoder(rid_rep)    # [B*gray, H]
        ctx_hash_code_re = self.ctx_hash_decoder(ctx_hash_code)    # [B, H]
        can_hash_code_re = self.can_hash_decoder(can_hash_code)    # [B*gray, H]

        # ===== KL Divergence ===== #
        kl_loss = F.cosine_similarity(ctx_hash_code_re, cid_rep, -1).mean()
        kl_loss += F.cosine_similarity(can_hash_code_re, rid_rep, -1).mean()
        kl_loss = - kl_loss * self.kl_loss_scale

        # ===== calculate quantization loss ===== #
        ctx_hash_code_h, can_hash_code_h = torch.sign(ctx_hash_code).detach(), torch.sign(can_hash_code).detach()
        quantization_loss = torch.norm(ctx_hash_code - ctx_hash_code_h, p=2, dim=1).mean() + torch.norm(can_hash_code - can_hash_code_h, p=2, dim=1).mean()
        
        # ===== calculate hash loss ===== #
        matrix = torch.matmul(ctx_hash_code, can_hash_code.T)    # [B, B*H] similarity matrix
        mask = torch.zeros_like(matrix)
        mask[torch.arange(batch_size), torch.arange(0, len(rid_rep), self.gray_num+1)] = 1.
        label_matrix = self.hash_code_size * mask
        hash_loss = torch.norm(matrix - label_matrix, p=2).mean() * self.hash_loss_scale
        
        # ===== calculate hamming distance for accuracy ===== #
        matrix = torch.matmul(ctx_hash_code_h, can_hash_code_h.t())    # [B, B]
        hamming_distance = 0.5 * (self.hash_code_size - matrix)    # hamming distance: ||b_i, b_j||_{H} = 0.5 * (K - b_i^Tb_j); [B, B]
        acc_num = (hamming_distance.min(dim=-1)[1] == torch.LongTensor(torch.arange(0, len(rid), self.gray_num+1)).cuda()).sum().item()
        acc = acc_num / batch_size

        return kl_loss, hash_loss, quantization_loss, acc
