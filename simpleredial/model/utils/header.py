from header import *
from collections import defaultdict
import torch
from torch.nn.parallel.data_parallel import DataParallel
from torch.nn.parallel.parallel_apply import parallel_apply
from torch.nn.parallel._functions import Scatter
import numpy as np
import math
import ipdb
import json
import re
import pickle
from tqdm import tqdm
from copy import deepcopy
import torch
import torch.nn as nn
from torch.nn import DataParallel
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import torch.nn.init as init
from torch.nn.utils.rnn import pad_sequence
import torch.optim as optim
from torch.optim import lr_scheduler
from collections import Counter, OrderedDict
from torch.nn.utils import clip_grad_norm_
import random
from transformers import BertTokenizer
from transformers import BertForSequenceClassification, BertModel, BertForPreTraining, EncoderDecoderModel, XLMRobertaModel, GPT2LMHeadModel, BertForMaskedLM
import transformers
from sklearn.metrics import label_ranking_average_precision_score
import argparse
import joblib
import faiss
import time
from torch.cuda.amp import autocast, GradScaler
from nlgeval import NLGEval
from bert_score import BERTScorer


# herits from huggingface models 
from transformers import BertPreTrainedModel
from transformers.models.bert.modeling_bert import BertEncoder, BertPooler, BertPreTrainingHeads, BertForPreTrainingOutput
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions, CausalLMOutputWithCrossAttentions, BaseModelOutputWithPastAndCrossAttentions, SequenceClassifierOutput
from transformers.file_utils import (
    ModelOutput,
    add_code_sample_docstrings,
    add_start_docstrings,
    add_start_docstrings_to_model_forward,
    replace_return_docstrings,
) 

from transformers.models.bert_generation.modeling_bert_generation import BertGenerationOnlyLMHead, BertGenerationPreTrainedModel
