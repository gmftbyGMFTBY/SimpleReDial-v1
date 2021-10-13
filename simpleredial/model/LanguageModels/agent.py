from model.utils import *
from dataloader.util_func import *

class LanguageModelsAgent(RetrievalBaseAgent):

    '''no train mode, only test'''
    
    def __init__(self, vocab, model, args):
        super(LanguageModelsAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.cuda_models = ['gpt2lm']
        if self.args['model'] in self.cuda_models:
            if torch.cuda.is_available():
                self.model.cuda()

    @torch.no_grad()
    def rerank(self, batches):
        '''rerank scores'''
        if self.args['model'] in self.cuda_models:
            self.model.eval()
        scores = []
        for batch in batches:
            # compatible for the predict function
            score = self.model.predict(batch)
            scores.append(score)
        return scores

    def load_model(self, path):
        pass

    @torch.no_grad()
    def inference(self, inf_iter, size=500000):
        self.model.eval()
        pbar = tqdm(inf_iter)
        ppls, texts = [], []
        for batch in pbar:
            ppl = self.model.module.predict(batch)
            ppls.extend(ppl)
            texts.extend(batch['candidates'])
        torch.save(
            (ppls, texts), 
            f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_ppl_{self.args["local_rank"]}.pt'
        )
