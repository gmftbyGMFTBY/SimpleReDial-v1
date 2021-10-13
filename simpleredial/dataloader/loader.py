from header import *
from .dual_bert_hier_dataloader import *
from .dual_bert_curriculum_learning_dataloader import *
from .horse_human_test_dataloader import *
from .time_evaluation_dataloader import *
from .fine_grained_test_dataloader import *
from .bert_mask_augmentation_dataloader import *
from .gpt2_dataloader import *
from .simcse_dataloader import *
from .post_train_dataloader import *
from .dual_bert_dataloader import *
from .dual_bert_unsup_dataloader import *
from .dual_bert_phrase_dataloader import *
from .dual_bert_pt_dataloader import *
from .dual_bert_full_dataloader import *
from .dual_bert_arxiv_dataloader import *
from .sa_bert_dataloader import *
from .bert_ft_dataloader import *
from .bert_ft_auxiliary_dataloader import *
from .bert_ft_compare_dataloader import *
from .inference_dataloader import *
from .inference_full_filter_dataloader import *
from .inference_phrase_dataloader import *
from .inference_full_dataloader import *
from .inference_ctx_dataloader import *

def load_dataset(args):
    if args['mode'] in ['train', 'test', 'valid']:
        dataset_name = args['models'][args['model']]['dataset_name']
        dataset_t = globals()[dataset_name]
    elif args['mode'] in ['inference']:
        # inference
        dataset_name = args['models'][args['model']]['inference_dataset_name']
        dataset_t = globals()[dataset_name]
    else:
        raise Exception(f'[!] Unknown mode: {args["mode"]}')

    if args['mode'] in ['inference']:
        path = f'{args["root_dir"]}/data/{args["dataset"]}/train.txt'
    else:
        path = f'{args["root_dir"]}/data/{args["dataset"]}/{args["mode"]}.txt'
    vocab = BertTokenizerFast.from_pretrained(args['tokenizer'])
    data = dataset_t(vocab, path, **args)

    if args['mode'] in ['train', 'inference']:
        sampler = torch.utils.data.distributed.DistributedSampler(data)
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate, sampler=sampler)
    else:
        iter_ = DataLoader(data, batch_size=args['batch_size'], collate_fn=data.collate)
        sampler = None
    try:
        if not os.path.exists(data.pp_path):
            data.save()
    except:
        pass
    return data, iter_, sampler
