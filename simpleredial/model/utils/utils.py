from .header import *


class PositionEmbedding(nn.Module):

    def __init__(self, d_model, dropout=0.5, max_len=512):
        super(PositionEmbedding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)
        pe = torch.zeros(max_len, d_model)    # [max_len, d_model]
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)    # [1, max_len]
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * (-math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0).transpose(0, 1)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:x.size(0), :]
        return self.dropout(x)


class GPT2LMIRModel(nn.Module):

    '''except for the LM output, also return the last hidden state for phrase-level information retrieval'''

    def __init__(self, model='uer/gpt2-base-chinese-cluecorpussmall'):
        super(GPT2LMIRModel, self).__init__()
        self.model = GPT2LMHeadModel.from_pretrained(model)

    def forward(self, ids, attn_mask):
        output = self.model(
            input_ids=ids,
            attention_mask=attn_mask,
            output_hidden_states=True,
        )
        lm_logits = output.logits
        hidden_state = output.hidden_states[-1]
        # lm_logits: [B, S, V]; hidden_state: [B, S, E]
        return lm_logits, hidden_state


class BertFullEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese', add_tokens=1):
        super(BertFullEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        # bert-fp checkpoint has the special token: [EOS]
        self.resize(add_tokens)

    def resize(self, num):
        self.model.resize_token_embeddings(self.model.config.vocab_size + num)

    def forward(self, ids, attn_mask, speaker_type_ids=None):
        # return: [B, S, E]
        embds = self.model(ids, attention_mask=attn_mask)[0]
        return embds


class TopKBertEmbedding(nn.Module):

    '''bert embedding with m query heads'''
    
    def __init__(self, model='bert-base-chinese', m=5, dropout=0.1):
        super(TopKBertEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.m = m
        # bert-fp checkpoint has the special token: [EOS]
        self.model.resize_token_embeddings(self.model.config.vocab_size + 1)
        self.queries = nn.Parameter(torch.randn(512, 768))    # [M, E] with maxium length 512
        self.proj_heads = nn.ModuleList([
            nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(768, 768)
            ) for _ in range(m)    
        ])

    def get_padding_mask_weight(self, attn_mask):
        weight = torch.where(attn_mask != 0, torch.zeros_like(attn_mask), torch.ones_like(attn_mask))
        weight = weight * -1e3    # [B, S]
        weight = weight.unsqueeze(1).repeat(1, self.m, 1)    # [B, M, S]
        return weight

    def forward(self, ids, attn_mask, speaker_type_ids=None):
        # embds = self.model(ids, attention_mask=attn_mask)[1]
        embds = self.model(ids, attention_mask=attn_mask)[0]     # [B, S, E]
        
        # [B, S, E] x [M, E] -> [B, S, E] x [E, M] -> [B, S, M] -> [B, M, S]
        queries = self.queries[:self.m, :]    # [M, E]
        scores = torch.matmul(embds, queries.t()).permute(0, 2, 1)
        scores /= np.sqrt(768)
        weight = self.get_padding_mask_weight(attn_mask)    # [B, M, S]
        scores += weight
        scores = F.softmax(scores, dim=-1)    # [B, M, S]

        # [B, M, S] x [B, S, E] -> [B, M, E]
        rep = torch.bmm(scores, embds)
        # project m times
        reps = []
        for idx, rep_ in enumerate(rep.permute(1, 0, 2)):
            # rep_: [B, E]
            # residual connection
            # rep_ = rep_ + self.proj_heads[idx](rep_)
            rep_ = self.proj_heads[idx](rep_)
            reps.append(rep_)
        reps = torch.stack(reps)    # [M, B, E]
        return reps


class BertMLEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese', add_tokens=1, topk_layer_num=3):
        super(BertMLEmbedding, self).__init__()
        self.topk_layer_num = topk_layer_num
        self.model = BertModel.from_pretrained(model)
        # bert-fp checkpoint has the special token: [EOS]
        self.resize(add_tokens)

    def resize(self, num):
        self.model.resize_token_embeddings(self.model.config.vocab_size + num)

    def forward(self, ids, attn_mask, speaker_type_ids=None):
        embds = self.model(ids, attention_mask=attn_mask, output_hidden_states=True)[2]    # 13 * [B, S, E]
        embds = [embd[:, 0, :] for embd in embds[-self.topk_layer_num:]]
        embds = torch.cat(embds, dim=-1)    # 3*[B, E] -> [B, 3*E]
        return embds


class BertEmbeddingWithWordEmbd(nn.Module):
    
    def __init__(self, model='bert-base-chinese', add_tokens=1):
        super(BertEmbeddingWithWordEmbd, self).__init__()
        self.model = BertModel.from_pretrained(model)
        self.resize(add_tokens)

    def resize(self, num):
        self.model.resize_token_embeddings(self.model.config.vocab_size + num)

    def forward(self, ids, attn_mask, word_embeddings=None):
        if word_embeddings is None:
            word_embeddings = self.model.embeddings.word_embeddings(ids)    # [B, S, 768]
            embds = self.model(
                input_ids=ids, 
                attention_mask=attn_mask,
            )[0]
            return embds[:, 0, :], word_embeddings
        else:
            embds = self.model(
                attention_mask=attn_mask,
                inputs_embeds=word_embeddings,
            )[0]
            return embds[:, 0, :]


class BertOAEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese', add_tokens=1, layer=2, dropout=0.1):
        super(BertOAEmbedding, self).__init__()
        self.model = BertModel.from_pretrained(model, add_pooling_layer=False)
        self.gru = nn.GRU(
            768, 
            768, 
            num_layers=layer, 
            dropout=(0 if layer == 1 else dropout)
        )
        self.fusion_head = nn.Sequential(
            nn.Linear(768*2, 768),
            nn.Tanh(),
            nn.Dropout(p=dropout),
            nn.Linear(768, 768)
        )
        # bert-fp checkpoint has the special token: [EOS]
        if add_tokens > 0:
            self.resize(add_tokens)

    def resize(self, num):
        self.model.resize_token_embeddings(self.model.config.vocab_size + num)

    def forward(self, ids, attn_mask, speaker_type_ids=None):
        embds = self.model(ids, attention_mask=attn_mask)[0]    # [B, S, E]
        # lengths
        lengths = []
        for mask in attn_mask:
            lengths.append(len(mask.nonzero()))
        embedded = nn.utils.rnn.pack_padded_sequence(
            embds.permute(1, 0, 2), 
            lengths,
            enforce_sorted=False
        )
        _, hidden = self.gru(embedded)    # [2, B, E]
        hidden = hidden.sum(axis=0)    # [B, E]
        embd = self.fusion_head(
            torch.cat([hidden, embds[:, 0, :]], dim=-1)        
        )
        embd = hidden + embds[:, 0, :]
        return embd


class SABertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese', add_tokens=1):
        super(SABertEmbedding, self).__init__()
        self.model = BertSAModel.from_pretrained(model, add_pooling_layer=False)
        # bert-fp checkpoint has the special token: [EOS]
        if add_tokens > 0:
            self.resize(add_tokens)

    def resize(self, num):
        self.model.resize_token_embeddings(self.model.config.vocab_size + num)

    def forward(self, ids, sids, tlids, attn_mask):
        embds = self.model(
            input_ids=ids,
            attention_mask=attn_mask,
            speaker_ids=sids,
        )[0]
        return embds[:, 0, :]


class BertEmbedding(nn.Module):
    
    def __init__(self, model='bert-base-chinese', add_tokens=1, load_param=True, hidden_dropout_ratio=0.1):
        super(BertEmbedding, self).__init__()
        if load_param:
            self.model = BertModel.from_pretrained(
                pretrained_model_name_or_path=model, 
                # add_pooling_layer=False,
                # hidden_dropout_prob=hidden_dropout_ratio,
                # attention_probs_dropout_prob=hidden_dropout_ratio,
            )
        else:
            config = BertConfig.from_pretrained(model)
            self.model = BertModel(config, add_pooling_layer=False)
        # bert-fp checkpoint has the special token: [EOS]
        if add_tokens > 0:
            self.resize(add_tokens)

    def resize(self, num):
        self.model.resize_token_embeddings(self.model.config.vocab_size + num)

    def forward(self, ids, attn_mask, speaker_type_ids=None):
        embds = self.model(ids, attention_mask=attn_mask)[0]
        return embds[:, 0, :]


# label smoothing loss
class HNLabelSmoothLoss(nn.Module):

    '''hard negative label smoothing'''
    
    def __init__(self, smoothing=0.0, hn_smoothing=0.0, topk=2):
        super(HNLabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
        self.hn_smoothing = hn_smoothing
        self.topk = topk
    
    def forward(self, input, target):
        index = [list(range(i*self.topk+1, i*self.topk+self.topk)) for i in range(len(input))]
        index = torch.LongTensor(index).to(input.device)

        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - self.topk)
        # weight for hard negative
        weight.scatter_(-1, index, self.hn_smoothing)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.hn_smoothing - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

# label smoothing loss
class LabelSmoothLoss(nn.Module):
    
    def __init__(self, smoothing=0.0):
        super(LabelSmoothLoss, self).__init__()
        self.smoothing = smoothing
    
    def forward(self, input, target):
        log_prob = F.log_softmax(input, dim=-1)
        weight = input.new_ones(input.size()) * \
            self.smoothing / (input.size(-1) - 1.)
        weight.scatter_(-1, target.unsqueeze(-1), (1. - self.smoothing))
        loss = (-weight * log_prob).sum(dim=-1).mean()
        return loss

'''https://github.com/taesunwhang/BERT-ResSel/blob/master/evaluation.py'''
def calculate_candidates_ranking(prediction, ground_truth, eval_candidates_num=10):
    total_num_split = len(ground_truth) / eval_candidates_num
    pred_split = np.split(prediction, total_num_split)
    gt_split = np.split(np.array(ground_truth), total_num_split)
    orig_rank_split = np.split(np.tile(np.arange(0, eval_candidates_num), int(total_num_split)), total_num_split)
    stack_scores = np.stack((gt_split, pred_split, orig_rank_split), axis=-1)
    
    rank_by_pred_l = []
    for i, stack_score in enumerate(stack_scores):
        rank_by_pred = sorted(stack_score, key=lambda x: x[1], reverse=True)
        rank_by_pred = np.stack(rank_by_pred, axis=-1)
        rank_by_pred_l.append(rank_by_pred[0])
    rank_by_pred = np.array(rank_by_pred_l)
    
    pos_index = []
    for sorted_score in rank_by_pred:
        curr_cand = []
        for p_i, score in enumerate(sorted_score):
            if int(score) == 1:
                curr_cand.append(p_i)
        pos_index.append(curr_cand)

    return rank_by_pred, pos_index, stack_scores


def logits_recall_at_k(pos_index, k_list=[1, 2, 5, 10]):
    # 1 dialog, 10 response candidates ground truth 1 or 0
    # prediction_score : [batch_size]
    # target : [batch_size] e.g. 1 0 0 0 0 0 0 0 0 0
    # e.g. batch : 100 -> 100/10 = 10
    num_correct = np.zeros([len(pos_index), len(k_list)])
    index_dict = dict()
    for i, p_i in enumerate(pos_index):
        index_dict[i] = p_i

    # case for douban : more than one correct answer case
    for i, p_i in enumerate(pos_index):
        if len(p_i) == 1 and p_i[0] >= 0:
            for j, k in enumerate(k_list):
                if p_i[0] + 1 <= k:
                    num_correct[i][j] += 1
        elif len(p_i) > 1:
            for j, k in enumerate(k_list):
                all_recall_at_k = []
                for cand_p_i in p_i:
                    if cand_p_i + 1 <= k:
                        all_recall_at_k.append(1)
                    else:
                        all_recall_at_k.append(0)
                num_correct[i][j] += np.mean(all_recall_at_k)
                # num_correct[i][j] += np.max(all_recall_at_k)

    return np.sum(num_correct, axis=0)

def logits_mrr(pos_index):
    mrr = []
    for i, p_i in enumerate(pos_index):
        if len(p_i) > 0 and p_i[0] >= 0:
            mrr.append(1 / (p_i[0] + 1))
        elif len(p_i) == 0:
            mrr.append(0)  # no answer

    return np.sum(mrr)

def precision_at_one(rank_by_pred):
    num_correct = [0] * rank_by_pred.shape[0]
    for i, sorted_score in enumerate(rank_by_pred):
        for p_i, score in enumerate(sorted_score):
            if p_i == 0 and int(score) == 1:
                num_correct[i] = 1
                break

    return np.sum(num_correct)

def mean_average_precision(pos_index):
    map = []
    for i, p_i in enumerate(pos_index):
        if len(p_i) > 0:
            all_precision = []
            for j, cand_p_i in enumerate(p_i):
                all_precision.append((j + 1) / (cand_p_i + 1))
            curr_map = np.mean(all_precision)
            map.append(curr_map)
        elif len(p_i) == 0:
            map.append(0)  # no answer

    return np.sum(map)

# ========== Metrics of the BERT-FP ========== #
class Metrics:

    def __init__(self):
        super(Metrics, self).__init__()
        # It depend on positive negative ratio 1:1 or 1:10
        self.segment = 10

    def __process_score_data(self, score_data):
        sessions = []
        one_sess = []
        i = 0
        for score, label in score_data:
            i += 1
            one_sess.append((score, label))
            if i % self.segment == 0:
                one_sess_tmp = np.array(one_sess)
                if one_sess_tmp[:, 1].sum() > 0:
                    # for douban (no positive cases)
                    sessions.append(one_sess)
                one_sess = []
        return sessions

    def __mean_average_precision(self, sort_data):
        count_1 = 0
        sum_precision = 0
        for index in range(len(sort_data)):
            if sort_data[index][1] == 1:
                count_1 += 1
                sum_precision += 1.0 * count_1 / (index+1)
        return sum_precision / count_1

    def __mean_reciprocal_rank(self, sort_data):
        sort_lable = [s_d[1] for s_d in sort_data]
        assert 1 in sort_lable
        return 1.0 / (1 + sort_lable.index(1))

    def __precision_at_position_1(self, sort_data):
        if sort_data[0][1] == 1:
            return 1
        else:
            return 0

    def __recall_at_position_k_in_10(self, sort_data, k):
        sort_label = [s_d[1] for s_d in sort_data]
        select_label = sort_label[:k]
        return 1.0 * select_label.count(1) / sort_label.count(1)

    def evaluation_one_session(self, data):
        np.random.shuffle(data)
        sort_data = sorted(data, key=lambda x: x[0], reverse=True)
        m_a_p = self.__mean_average_precision(sort_data)
        m_r_r = self.__mean_reciprocal_rank(sort_data)
        p_1   = self.__precision_at_position_1(sort_data)
        r_1   = self.__recall_at_position_k_in_10(sort_data, 1)
        r_2   = self.__recall_at_position_k_in_10(sort_data, 2)
        r_5   = self.__recall_at_position_k_in_10(sort_data, 5)
        return m_a_p, m_r_r, p_1, r_1, r_2, r_5

    def evaluate_all_metrics(self, data):
        '''data is a list of double item tuple: [(score, label), ...]'''
        sum_m_a_p = 0
        sum_m_r_r = 0
        sum_p_1 = 0
        sum_r_1 = 0
        sum_r_2 = 0
        sum_r_5 = 0

        sessions = self.__process_score_data(data)
        total_s = len(sessions)
        for session in sessions:
            m_a_p, m_r_r, p_1, r_1, r_2, r_5 = self.evaluation_one_session(session)
            sum_m_a_p += m_a_p
            sum_m_r_r += m_r_r
            sum_p_1 += p_1
            sum_r_1 += r_1
            sum_r_2 += r_2
            sum_r_5 += r_5

        return (sum_m_a_p/total_s,
                sum_m_r_r/total_s,
                  sum_p_1/total_s,
                  sum_r_1/total_s,
                  sum_r_2/total_s,
                  sum_r_5/total_s)

# ========== PageRank Scorer ========== #
class PageRank:

    def __init__(self, vertices, graph, alpha=0.9, init_num=1., iter_num=100):
        self.n = vertices
        self.alpha = alpha
        self.init_num = init_num
        self.iter_num = iter_num
        self.graph = graph

    def iter(self):
        # init
        raw_g = [[0 for _ in range(self.n)] for _ in range(self.n)]
        for i, j in self.graph:
            raw_g[i][j] = 1
        adj = np.full([self.n, self.n], raw_g, dtype=float)
        # normalization
        for i in range(self.n):
            if adj[i].sum() > 0:
                adj[i] /= adj[i].sum()

        pr = np.full([1, self.n], self.init_num, dtype=float)
        jump = np.full(
            [2, 1], 
            [
                [self.alpha], 
                [1-self.alpha]
            ], 
            dtype=float
        )
        for _ in range(self.iter_num):
            pr = np.dot(pr, adj)    # [1, n]
            
            pr_jump = np.full([self.n, 2], [[0, 1/self.n]])
            pr_jump[:, :-1] = pr.transpose()

            pr = np.dot(pr_jump, jump)    # [n, 1]

            pr = pr.transpose()    # [1, n]
            pr = pr / pr.sum()
        return pr.squeeze()    # [n]


# ========== Topo Sort ========= #
class Graph: 

    '''0 for not searched; 1 for searching; 2 for searched'''

    def __init__(self, vertices): 
        self.graph = defaultdict(list) 
        self.V = vertices
        # whether the graph have the loop
        self.valid = True
  
    def addEdge(self,u,v): 
        self.graph[u].append(v) 
  
    def topologicalSortUtil(self, v, visited, stack): 
        visited[v] = 1    #  set searching status
        for i in self.graph[v]: 
            if visited[i] == 0: 
                self.topologicalSortUtil(i, visited, stack) 
                if self.valid is False:
                    return
            elif visited[i] == 1:
                self.valid = False
                return
        visited[v] = 2    # set searched status
        stack.insert(0, v) 
  
    def topologicalSort(self): 
        visited = [0] * self.V 
        stack = [] 
        for i in range(self.V): 
            if self.valid and visited[i] == 0: 
                self.topologicalSortUtil(i, visited, stack) 
        # the valid status can be accessed by self.valid
        return stack

# ========== Model State Dict Adapter ========= #
class CheckpointAdapter:

    '''convert the from named paramters to target named paramters'''

    def __init__(self):
        self.prefix_list = ['bert_model', 'model']
        self.maybe_missing_list = [
            'embeddings.position_ids', 
            'embeddings.speaker_embeddings.weight', 
            'model.embeddings.position_ids', 
            'model.bert.embeddings.position_ids', 
            # the followings are the BERTDualO2MTopKEmbedding extra parameters
            'queries',
        ]

    def clean(self, name):
        for prefix in self.prefix_list:
            name = name.lstrip(f'{prefix}.')
        return name

    def init(self, from_np, target_np):
        self.mapping, self.missing, self.unused = self._init(from_np, target_np)
        self.show_inf()

    def _init(self, from_np, target_np):
        def _target_to_from():
            mapping = {}
            missing = []
            for tname in target_np:
                for fname in from_np:
                    if self.clean(tname) in self.clean(fname):
                        mapping[fname] = tname
                        break
                else:
                    missing.append(tname)
            return mapping, missing

        def _from_to_target():
            mapping = {}
            for fname in from_np:
                for tname in target_np:
                    if self.clean(fname) in self.clean(tname):
                        mapping[fname] = tname
                        break
            missing = list(set(target_np) - set(mapping.values()))
            return mapping, missing
        
        def _judge(collected_paramters, missing):
            try:
                assert len(collected_paramters) > 0
                assert len(collected_paramters) == len(set(collected_paramters))
                for i in missing:
                    for k in self.maybe_missing_list:
                        if k in i:
                            break
                    else:
                        raise Exception(f'[!] ERROR find missing parameters: {i}')
            except:
                return False
            return True
        from_np, target_np = list(from_np), list(target_np)
        mapping, missing = _target_to_from()
        collected_paramters = list(mapping.values())
        if _judge(collected_paramters, missing):
            unused = list(set(collected_paramters) - set(target_np))
            return mapping, missing, unused
        mapping, missing = _from_to_target()
        collected_paramters = list(mapping.values())
        if _judge(collected_paramters, missing):
            unused = list(set(collected_paramters) - set(target_np))
            return mapping, missing, unused
        # show the error log
        if len(missing) > 0:
            print(f'[!] !!!!! Find missing parameters !!!!!')
            for i in missing:
                print(f'   - ', i)
        raise Exception(f'[!] Load checkpoint failed')

    def show_inf(self):
        if len(self.unused) > 0:
            print(f'[!] Find unused parameters:')
            for i in self.unused:
                print(f'   - ', i)

    def convert(self, from_state_dict):
        new_state_dict = OrderedDict()
        for k, v in from_state_dict.items():
            if k in self.mapping:
                k_ = self.mapping[k]
                new_state_dict[k_] = v
        for i in self.missing:
            # missing parameters are the position ids
            if i in ['model.bert.embeddings.position_ids', 'embeddings.position_ids', 'model.embeddings.position_ids']:
                new_state_dict[i] = torch.arange(512).expand((1, -1))
            # elif i in ['embeddings.speaker_embedding.weight']:
            #     new_state_dict[i] = nn.Embedding(2, 768)
            elif i in ['queries']:
                new_state_dict[i] = torch.randn(512, 768)
        return new_state_dict

def distributed_collect(cid_rep, rid_rep):
    # all_gather collects the samples of other processes for distributed training
    # More details can be found in this link:
    #     https://github.com/princeton-nlp/SimCSE/blob/main/simcse/models.py#L170
    cid_reps = [torch.zeros_like(cid_rep) for _ in range(dist.get_world_size())]
    rid_reps = [torch.zeros_like(rid_rep) for _ in range(dist.get_world_size())]
    dist.all_gather(tensor_list=cid_reps, tensor=cid_rep.contiguous())
    dist.all_gather(tensor_list=rid_reps, tensor=rid_rep.contiguous())
    # only the current process's tensors have the gradient
    cid_reps[dist.get_rank()] = cid_rep
    rid_reps[dist.get_rank()] = rid_rep
    cid_reps = torch.cat(cid_reps, dim=0)
    rid_reps = torch.cat(rid_reps, dim=0)
    return cid_reps, rid_reps

# FGM
class FGM:

    def __init__(self, model):
        self.model = model
        self.backup = {}

    def attack(self, epsilon=1., emb_name='embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name:
                # ipdb.set_trace()
                self.backup[name] = param.data.clone()
                norm = torch.norm(param.grad)
                if norm != 0 and not torch.isnan(norm):
                    r_at = epsilon * param.grad / norm
                    param.data.add_(r_at)

    def restore(self, emb_name='embeddings.'):
        for name, param in self.model.named_parameters():
            if param.requires_grad and emb_name in name: 
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

# ========== Speaker-aware BERT Model ========== #
class BertSAEmbeddings(nn.Module):

    '''speaker-aware and turn-level embeddings are added'''

    def __init__(self, config):
        super().__init__()
        self.word_embeddings = nn.Embedding(config.vocab_size, config.hidden_size, padding_idx=config.pad_token_id)
        self.position_embeddings = nn.Embedding(config.max_position_embeddings, config.hidden_size)
        self.token_type_embeddings = nn.Embedding(config.type_vocab_size, config.hidden_size)
        self.speaker_embeddings = nn.Embedding(2, 768)

        # self.LayerNorm is not snake-cased to stick with TensorFlow model variable name and be able to load
        # any TensorFlow checkpoint file
        self.LayerNorm = nn.LayerNorm(config.hidden_size, eps=config.layer_norm_eps)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)
        # position_ids (1, len position emb) is contiguous in memory and exported when serialized
        self.position_embedding_type = getattr(config, "position_embedding_type", "absolute")
        self.register_buffer("position_ids", torch.arange(config.max_position_embeddings).expand((1, -1)))
        # make sure the torch version > 1.6.0
        self.register_buffer(
            "token_type_ids",
            torch.zeros(self.position_ids.size(), dtype=torch.long, device=self.position_ids.device),
            persistent=False,
        )

    def forward(
        self, input_ids=None, token_type_ids=None, position_ids=None, inputs_embeds=None, past_key_values_length=0, speaker_ids=None,
    ):
        if input_ids is not None:
            input_shape = input_ids.size()
        else:
            input_shape = inputs_embeds.size()[:-1]

        seq_length = input_shape[1]

        if position_ids is None:
            position_ids = self.position_ids[:, past_key_values_length : seq_length + past_key_values_length]

        # Setting the token_type_ids to the registered buffer in constructor where it is all zeros, which usually occurs
        # when its auto-generated, registered buffer helps users when tracing the model without passing token_type_ids, solves
        # issue #5664
        if token_type_ids is None:
            if hasattr(self, "token_type_ids"):
                buffered_token_type_ids = self.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(input_shape[0], seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=self.position_ids.device)

        if inputs_embeds is None:
            inputs_embeds = self.word_embeddings(input_ids)
        token_type_embeddings = self.token_type_embeddings(token_type_ids)
        embeddings = inputs_embeds
        if speaker_ids is not None:
            speaker_embeddings = self.speaker_embeddings(speaker_ids)
            embeddings += speaker_embeddings
        embeddings += token_type_embeddings
        if self.position_embedding_type == "absolute":
            position_embeddings = self.position_embeddings(position_ids)
            embeddings += position_embeddings
        embeddings = self.LayerNorm(embeddings)
        embeddings = self.dropout(embeddings)
        return embeddings


class BertSAModel(BertPreTrainedModel):

    def __init__(self, config, add_pooling_layer=True):
        super().__init__(config)
        self.config = config

        self.embeddings = BertSAEmbeddings(config)
        self.encoder = BertEncoder(config)

        self.pooler = BertPooler(config) if add_pooling_layer else None

        self.init_weights()

    def get_input_embeddings(self):
        return self.embeddings.word_embeddings

    def set_input_embeddings(self, value):
        self.embeddings.word_embeddings = value

    def _prune_heads(self, heads_to_prune):
        for layer, heads in heads_to_prune.items():
            self.encoder.layer[layer].attention.prune_heads(heads)

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        speaker_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        past_key_values=None,
        use_cache=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        output_attentions = output_attentions if output_attentions is not None else self.config.output_attentions
        output_hidden_states = (
            output_hidden_states if output_hidden_states is not None else self.config.output_hidden_states
        )
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        if self.config.is_decoder:
            use_cache = use_cache if use_cache is not None else self.config.use_cache
        else:
            use_cache = False

        if input_ids is not None and inputs_embeds is not None:
            raise ValueError("You cannot specify both input_ids and inputs_embeds at the same time")
        elif input_ids is not None:
            input_shape = input_ids.size()
        elif inputs_embeds is not None:
            input_shape = inputs_embeds.size()[:-1]
        else:
            raise ValueError("You have to specify either input_ids or inputs_embeds")

        batch_size, seq_length = input_shape
        device = input_ids.device if input_ids is not None else inputs_embeds.device

        # past_key_values_length
        past_key_values_length = past_key_values[0][0].shape[2] if past_key_values is not None else 0

        if attention_mask is None:
            attention_mask = torch.ones(((batch_size, seq_length + past_key_values_length)), device=device)

        if token_type_ids is None:
            if hasattr(self.embeddings, "token_type_ids"):
                buffered_token_type_ids = self.embeddings.token_type_ids[:, :seq_length]
                buffered_token_type_ids_expanded = buffered_token_type_ids.expand(batch_size, seq_length)
                token_type_ids = buffered_token_type_ids_expanded
            else:
                token_type_ids = torch.zeros(input_shape, dtype=torch.long, device=device)

        # We can provide a self-attention mask of dimensions [batch_size, from_seq_length, to_seq_length]
        # ourselves in which case we just need to make it broadcastable to all heads.
        extended_attention_mask: torch.Tensor = self.get_extended_attention_mask(attention_mask, input_shape, device)

        # If a 2D or 3D attention mask is provided for the cross-attention
        # we need to make broadcastable to [batch_size, num_heads, seq_length, seq_length]
        if self.config.is_decoder and encoder_hidden_states is not None:
            encoder_batch_size, encoder_sequence_length, _ = encoder_hidden_states.size()
            encoder_hidden_shape = (encoder_batch_size, encoder_sequence_length)
            if encoder_attention_mask is None:
                encoder_attention_mask = torch.ones(encoder_hidden_shape, device=device)
            encoder_extended_attention_mask = self.invert_attention_mask(encoder_attention_mask)
        else:
            encoder_extended_attention_mask = None

        # Prepare head mask if needed
        # 1.0 in head_mask indicate we keep the head
        # attention_probs has shape bsz x n_heads x N x N
        # input head_mask has shape [num_heads] or [num_hidden_layers x num_heads]
        # and head_mask is converted to shape [num_hidden_layers x batch x num_heads x seq_length x seq_length]
        head_mask = self.get_head_mask(head_mask, self.config.num_hidden_layers)

        embedding_output = self.embeddings(
            input_ids=input_ids,
            position_ids=position_ids,
            token_type_ids=token_type_ids,
            inputs_embeds=inputs_embeds,
            past_key_values_length=past_key_values_length,
            speaker_ids=speaker_ids,
        )
        encoder_outputs = self.encoder(
            embedding_output,
            attention_mask=extended_attention_mask,
            head_mask=head_mask,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_extended_attention_mask,
            past_key_values=past_key_values,
            use_cache=use_cache,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = encoder_outputs[0]
        pooled_output = self.pooler(sequence_output) if self.pooler is not None else None

        if not return_dict:
            return (sequence_output, pooled_output) + encoder_outputs[1:]

        return BaseModelOutputWithPoolingAndCrossAttentions(
            last_hidden_state=sequence_output,
            pooler_output=pooled_output,
            past_key_values=encoder_outputs.past_key_values,
            hidden_states=encoder_outputs.hidden_states,
            attentions=encoder_outputs.attentions,
            cross_attentions=encoder_outputs.cross_attentions,
        )


def cosine_distance(x1, x2):
    # x1/x2: [B, E]
    sim = F.cosine_similarity(x1, x2, dim=-1)
    sim = (1 + sim) / 2    # range from 0 to 1
    return sim    # [B]
