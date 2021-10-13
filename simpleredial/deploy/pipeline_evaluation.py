from header import *
from config import *
from dataloader import *
from inference import Searcher
from es.es_utils import *
from model import Metrics
from .utils import *
from .rerank import *
from .recall import *


class PipelineEvaluationAgent:

    '''Evaluation:
        R@1000, R@500, R@100, R@50, MRR
    '''

    def __init__(self, args):
        self.args = args
        recall_args, rerank_args = args['recall'], args['rerank']
        self.recallagent = RecallAgent(recall_args)
        self.rerankagent = RerankAgent(rerank_args)

        # collection for calculating the metrics
        self.collection = {
            'R@1000': [],
            'R@500': [],
            'R@100': [],
            'R@50': [],
            'MRR': [],
        }

    @timethis
    def work(self, batch, topk=None):
        assert len(batch) == 1
        # recall
        candidates, recall_t = self.recallagent.work(batch, topk=topk)
        
        # re-packup
        r = [i['text'] for i in candidates[0]]
        rerank_batch = [{'context': batch[0]['str'], 'candidates': r}]

        # rerank
        scores, rerank_t = self.rerankagent.work(rerank_batch)

        # packup
        score, candidate = scores[0], candidates[0]
        sort_index = np.argsort(score)[::-1]
        score = [score[i] for i in sort_index]
        candidate = [candidate[i]['text'] for i in sort_index]
        ground_truths = batch[0]['ground-truth']

        # update the evaluation and the collector
        # recall metrics
        for idx in [1000, 500, 100, 50]:
            counter = 0
            for g in ground_truths:
                if g in candidate[:idx]:
                    counter += 1
            m = 0 if len(ground_truths) == 0 else counter/len(ground_truths)
            self.collection[f'R@{idx}'].append(m)
        # mrr 
        count_1 = 0
        sum_p = 0
        for g in ground_truths:
            if g in candidate:
                count_1 += 1
                sum_p += 1.0 * count_1 / (candidate.index(g) + 1)
        mrr = sum_p / count_1 if count_1 > 0 else 0
        self.collection['MRR'].append(mrr)

        return [candidate[0]], [mrr], recall_t, rerank_t


