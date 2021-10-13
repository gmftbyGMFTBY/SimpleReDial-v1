from header import *
from config import *
from dataloader import *
from inference import Searcher
from es.es_utils import *
from .utils import *
from .rerank import *
from .recall import *


class PipelineAgent:

    def __init__(self, args):
        self.args = args
        recall_args, rerank_args = args['recall'], args['rerank']
        self.recallagent = RecallAgent(recall_args)
        self.rerankagent = RerankAgent(rerank_args)

    @timethis
    def work(self, batch, topk=None):
        # recall
        topk = topk if topk else self.args['recall']['topk']
        candidates, recall_t = self.recallagent.work(batch, topk=topk)
        
        # re-packup
        contexts = [i['str'] for i in batch]
        rerank_batch = []
        for c, r in zip(contexts, candidates):
            r = [i['text'] for i in r]
            rerank_batch.append({'context': c, 'candidates': r})

        # rerank
        scores, rerank_t = self.rerankagent.work(rerank_batch)

        # packup
        responses = []
        for score, candidate in zip(scores, candidates):
            idx = np.argmax(score)
            responses.append(candidate[idx]['text'])
        return responses, recall_t, rerank_t
