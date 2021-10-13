from model.utils import *
from dataloader.util_func import *

class AugmentationAgent(RetrievalBaseAgent):
    
    def __init__(self, vocab, model, args):
        super(AugmentationAgent, self).__init__()
        self.args = args
        self.vocab, self.model = vocab, model
        self.pad, self.sep, self.eos, self.cls = self.vocab.convert_tokens_to_ids(['[PAD]', '[SEP]', '[EOS]', '[CLS]'])

        # open the test save scores file handler
        pretrained_model_name = self.args['pretrained_model'].replace('/', '_')
        path = f'{self.args["root_dir"]}/rest/{self.args["dataset"]}/{self.args["model"]}/scores_log_{pretrained_model_name}.txt'
        self.log_save_file = open(path, 'w')
        if torch.cuda.is_available():
            self.model.cuda()
        self.set_optimizer_scheduler_ddp()
        self.show_parameters(self.args)

    @torch.no_grad()
    def inference(self, inf_iter, size=500000):
        self.model.eval()
        contexts, responses, results = [], [], []
        for batch in tqdm(inf_iter):
            response = batch['response']
            context = batch['context']
            rest = self.model(batch)
            assert len(rest) == len(context) == len(response)
            contexts.extend(context)
            responses.extend(response)
            results.extend(rest)
        torch.save((contexts, responses, results), f'{self.args["root_dir"]}/data/{self.args["dataset"]}/inference_bert_mask_da_{self.args["local_rank"]}.pt')

    def load_model(self, path):
        state_dict = torch.load(path, map_location=torch.device('cpu'))
        missing, unexcept = self.model.module.load_state_dict(state_dict, strict=False)
        print(f'[!] load PLMs from {path}')
