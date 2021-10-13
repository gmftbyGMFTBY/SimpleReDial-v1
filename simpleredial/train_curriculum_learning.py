from header import *
from dataloader import *
from model import *
from config import *


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--local_rank', type=int)
    return parser.parse_args()


def obtain_steps_parameters(train_data, args):
    # multiple 2 means the 2 stage of the curriculumn training
    args['total_step'] = 2 * len(train_data) * args['epoch'] // args['batch_size'] // (args['multi_gpu'].count(',') + 1)
    args['warmup_step'] = int(args['warmup_ratio'] * args['total_step'])


def main(**args):
    torch.cuda.set_device(args['local_rank'])
    torch.distributed.init_process_group(backend='nccl', init_method='env://')

    # test set configuration
    test_args = deepcopy(args)
    test_args['mode'] = 'test'

    args['mode'] = 'train'
    config = load_config(args)
    args.update(config)
    if args['model'] in args['no_train_models']:
        raise Exception(f'[!] {args["model"]} is not allowed to be trained')
    
    # check
    if args['model'] not in args['curriculum_learning_models']:
        raise Exception(f'[!] {args["model"]} should not be trained with the curriculum learning strategy')

    # load dataset
    train_data, train_iter, sampler = load_dataset(args)

    if args['model'] not in args['no_test_models']:
        config = load_config(test_args)
        test_args.update(config)

        if args['valid_during_training']:
            # valid set for training
            test_args['mode'] = 'valid'
        test_data, test_iter, _ = load_dataset(test_args)
    else:
        test_iter = None
    
    # set seed
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    obtain_steps_parameters(train_data, args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    if args['local_rank'] == 0:
        sum_writer = SummaryWriter(
            log_dir=f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/{args["version"]}',
            comment=pretrained_model_name,
        )
    else:
        sum_writer = None
    batch_num = 0
    # 1st: easy in-batch negative mining
    for epoch_i in range(args['epoch']):
        sampler.set_epoch(epoch_i)    # shuffle for DDP
        nb = agent.train_model(
            train_iter, 
            test_iter,
            recoder=sum_writer,
            idx_=epoch_i,
            hard=False,
            whole_batch_num=batch_num,
        )
        batch_num += nb
    # 2nd: hard negative mining
    print(f'[!] BEGIN the 2nd curriculum training stage')
    train_iter.dataset.mode = 'hard'
    agent.best_test = None
    for epoch_i in range(args['epoch'], args['epoch'] * 2):
        sampler.set_epoch(epoch_i)    # shuffle for DDP
        nb = agent.train_model(
            train_iter, 
            test_iter,
            recoder=sum_writer,
            idx_=epoch_i,
            hard=True,
            whole_batch_num=batch_num,
        )
        batch_num += nb
    if sum_writer:
        sum_writer.close()
    if agent.best_test is None:
        pmn = args['pretrained_model'].replace('/', '_')
        save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pmn}_{args["version"]}.pt'
        agent.save_model(save_path)

if __name__ == "__main__":
    args = parser_args()
    args = vars(args)
    main(**args)
