from header import *
from .utils import *
from .util_func import *

class LMDataset(Dataset):
    
    def __init__(self, vocab, path, **args):
        self.args = args
        self.data = read

    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        bundle = self.data[i]
        return bundle['sentence']

    def collate(self, batch):
        sentences = batch
        return {
            'sentences': sentences
        }
