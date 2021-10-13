from header import *
from model import *
from config import *
from dataloader import *
from .utils import *


class RerankAgent:

    def __init__(self, args):
        if args['model'] is None:
            # donot rerank
            pass
        else:
            self.agent = load_model(args) 
            pretrained_model_name = args['pretrained_model'].replace('/', '_')
            save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
            self.agent.load_model(save_path)
        self.args = args

    @timethis
    def work(self, batches):
        if self.args['model'] is None:
            # remain the order
            scores = []
            for batch in batches:
                l = len(batch['candidates'])
                scores.append(list(range(l, 0, -1)))
        else:
            scores = self.agent.rerank(batches)
        return scores
