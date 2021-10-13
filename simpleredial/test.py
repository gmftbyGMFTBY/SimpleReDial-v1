from header import *
from dataloader import *
from model import *
from config import *
from inference import *
from es import *

'''
Test script:
    1. test the rerank performance [rerank]
    2. test the recall performance [recall]
    3. test the es recall performance [es_recall]
    4. test the comparison relationship among candidates [compare]

If you use the [recall], make sure the inference.sh has already been done
'''


def parser_args():
    parser = argparse.ArgumentParser(description='train parameters')
    parser.add_argument('--dataset', default='ecommerce', type=str)
    parser.add_argument('--model', type=str)
    parser.add_argument('--multi_gpu', type=str, default=None)
    parser.add_argument('--mode', type=str, default='test')
    parser.add_argument('--recall_mode', type=str, default='q-r')
    parser.add_argument('--file_tags', type=str, default='22335,22336')
    parser.add_argument('--log', action='store_true', dest='log')
    parser.add_argument('--no-log', action='store_false', dest='log')
    parser.add_argument('--candidate_size', type=int, default=100)
    return parser.parse_args()


def prepare_inference(**args):
    '''prepare the dataloader and the faiss index for recall test'''
    # use test mode args load test dataset and model
    inf_args = deepcopy(args)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    agent = load_model(args)
    # print('test', args)
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)

    test_data, test_iter, _ = load_dataset(args)

    # ===== use inference args ===== #
    inf_args['mode'] = 'inference'
    config = load_config(inf_args)
    inf_args.update(config)
    # print(f'inference', inf_args)

    # load faiss index
    model_name = inf_args['model']
    pretrained_model_name = inf_args['pretrained_model']
    if inf_args['recall_mode'] == 'q-q':
        q_q = True
        faiss_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_q_q_faiss.ckpt'        
        corpus_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_q_q_corpus.ckpt'        
    else:
        q_q = False
        faiss_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_faiss.ckpt'        
        corpus_ckpt_path = f'{inf_args["root_dir"]}/data/{inf_args["dataset"]}/{model_name}_{pretrained_model_name}_corpus.ckpt'        
    searcher = Searcher(inf_args['index_type'], dimension=inf_args['dimension'], q_q=q_q)
    searcher.load(faiss_ckpt_path, corpus_ckpt_path)
    print(f'[!] load faiss over')
    return test_iter, inf_args, searcher, agent


def main_compare(**args):
    '''compare mode applications:
        1. replace with the human evaluation
        2. rerank agent for rerank agent'''
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)

    results = agent.compare_evaluation(test_iter)

    log_file_path = f'{args["root_dir"]}/data/{args["dataset"]}/test_compare_log.txt'
    avg_scores = []
    with open(log_file_path, 'w') as f:
        for item in results:
            f.write(f'[Context   ] {item["context"]}\n')
            f.write(f'[Response-1] {item["responses"][0]}\n')
            f.write(f'[Response-2] {item["responses"][1]}\n')
            f.write(f'[Comp-score] {item["score"]}\n\n')
            avg_scores.append(item['score'])
    avg_score = round(np.mean(avg_scores), 2)
    print(f'[!] Write the log into {log_file_path}')
    print(f'[!] Average Compare Scores: {avg_score}')


def main_rerank(**args):
    '''whether to use the rerank model'''
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)
    # print('test', args)

    if args['model'] in args['no_test_models']:
        print(f'[!] model {args["model"]} doesn"t support test')
        return
    
    if args['rank']:
        new_args['model'] = args['rank']
        config = load_config(new_args)
        new_args.update(config)
        print(f'[!] RERANK AGENT IS USED')
    else:
        new_args = None
        print(f'[!] RERANK AGENT IS NOT USED')

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)

    # rerank model
    if new_args:
        rerank_agent = load_model(new_args)
        save_path = f'{new_args["root_dir"]}/ckpt/{new_args["dataset"]}/{new_args["model"]}/best_{pretrained_model_name}.pt'
        rerank_agent.load_model(save_path)
        print(f'[!] load rank order agent from: {save_path}')
    else:
        rerank_agent = None

    bt = time.time()
    outputs = agent.test_model(test_iter, print_output=True, rerank_agent=rerank_agent)
    cost_time = time.time() - bt
    cost_time *= 1000    # ms
    cost_time /= len(test_iter)

    with open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_result_rerank_{pretrained_model_name}_{args["version"]}.txt', 'w') as f:
        for key, value in outputs.items():
            print(f'{key}: {value}', file=f)
        print(f'Cost-Time: {round(cost_time, 2)} ms', file=f)
    for key, value in outputs.items():
        print(f'{key}: {value}')
    print(f'Cost-Time: {round(cost_time, 2)} ms')


def main_es_recall(**args):
    '''test the recall with the faiss index'''
    # use test mode args load test dataset and model
    inf_args = deepcopy(args)
    args['mode'] = 'test'
    config = load_config(args)
    args.update(config)
    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)

    # inference
    inf_args['mode'] = 'inference'
    config = load_config(inf_args)
    inf_args.update(config)

    searcher = ESSearcher(
        f'{inf_args["dataset"]}_{inf_args["recall_mode"]}', 
        q_q=True if inf_args['recall_mode'] == 'q-q' else False
    )

    # test recall (Top-20, Top-100)
    pbar = tqdm(test_iter)
    counter, acc = 0, 0
    cost_time = []
    log_collector = []
    for batch in pbar:
        if 'ids' in batch:
            context = agent.convert_to_text(batch['ids'])
        elif 'context' in batch:
            context = batch['context']
        else:
            raise Exception(f'[!] Error during test es recall')

        bt = time.time()
        rest = searcher.search(context, topk=inf_args['topk'])
        et = time.time()
        cost_time.append(et - bt)

        # print output
        log_collector.append({
            'context': context,
            'rest': rest,
        })

        gt_candidate = batch['text']
        if len(gt_candidate) == 0:
            continue
        for text in gt_candidate:
            if text in rest:
                acc += 1
                break
        counter += 1
        pbar.set_description(f'[!] Top-{inf_args["topk"]}: {round(acc/counter, 4)}')

    topk_metric = round(acc/counter, 4)
    avg_time = round(np.mean(cost_time)*1000, 2)    # ms
    pretrained_model_name = inf_args['pretrained_model'].replace('/', '_')
    print(f'[!] Top-{inf_args["topk"]}: {topk_metric}')
    print(f'[!] Average Times: {avg_time} ms')
    with open(f'{inf_args["root_dir"]}/rest/{inf_args["dataset"]}/{inf_args["model"]}/test_result_es_recall_{pretrained_model_name}.txt', 'w') as f:
        print(f'Top-{inf_args["topk"]}: {topk_metric}', file=f)
        print(f'Average Times: {avg_time} ms', file=f)
    if args['log']:
        with open(f'{inf_args["root_dir"]}/rest/{inf_args["dataset"]}/{inf_args["model"]}/recall_es_log.txt', 'w') as f:
            for item in log_collector:
                f.write(f'[Context] {item["context"]}\n')
                # DO NOT SAVE ALL THE RESULTS INTO THE LOG FILE (JUST TOP-10)
                for neg in item['rest'][:10]:
                    f.write(f'{neg}\n')
                f.write('\n')


def main_recall(**args):
    '''test the recall with the faiss index'''
    # use test mode args load test dataset and model
    test_iter, inf_args, searcher, agent = prepare_inference(**args)

    # test recall (Top-20, Top-100)
    pbar = tqdm(test_iter)
    counter, acc = 0, 0
    cost_time = []
    log_collector = []
    for batch in pbar:
        if 'ids' in batch:
            ids = batch['ids'].unsqueeze(0)
            ids_mask = torch.ones_like(ids)
        elif 'context' in batch:
            ids, ids_mask = agent.model.totensor([batch['context']], ctx=True)
        else:
            raise Exception(f'[!] process test dataset error')
        vector = agent.model.get_ctx(ids, ids_mask)   # [E]

        if not searcher.binary:
            vector = vector.cpu().numpy()

        bt = time.time()
        rest = searcher._search(vector, topk=inf_args['topk'])[0]
        et = time.time()
        cost_time.append(et - bt)

        #
        if 'context' in batch:
            context = batch['context']
        elif 'ids' in batch:
            context = agent.convert_to_text(batch['ids'])
        log_collector.append({
            'context': context,
            'rest': rest
        })

        gt_candidate = batch['text']
        if len(gt_candidate) == 0:
            continue
        for text in gt_candidate:
            if text in rest:
                acc += 1
                break
        counter += 1
        pbar.set_description(f'[!] Top-{inf_args["topk"]}: {round(acc/counter, 4)}')

    topk_metric = round(acc/counter, 4)
    avg_time = round(np.mean(cost_time)*1000, 2)    # ms
    pretrained_model_name = inf_args['pretrained_model'].replace('/', '_')
    print(f'[!] Top-{inf_args["topk"]}: {topk_metric}')
    print(f'[!] Average Times: {avg_time} ms')
    with open(f'{inf_args["root_dir"]}/rest/{inf_args["dataset"]}/{inf_args["model"]}/test_result_recall_{pretrained_model_name}.txt', 'w') as f:
        print(f'Top-{inf_args["topk"]}: {topk_metric}', file=f)
        print(f'Average Times: {avg_time} ms', file=f)
    if args['log']:
        model_name = args['model']
        with open(f'{inf_args["root_dir"]}/rest/{inf_args["dataset"]}/{inf_args["model"]}/recall_{model_name}_log.txt', 'w') as f:
            for item in log_collector:
                f.write(f'[Context] {item["context"]}\n')
                # DO NOT SAVE ALL THE RESULTS INTO THE LOG FILE (JUST TOP-10)
                for neg in item['rest'][:10]:
                    f.write(f'{neg}\n')
                f.write('\n')

def main_horse_human(**args):
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    if args['model'] in args['no_test_models']:
        print(f'[!] model {args["model"]} doesn"t support test')
        return
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}_{args["version"]}.pt'
    agent.load_model(save_path)
    collections = agent.test_model_horse_human(test_iter, print_output=True)
    ndcg_3, ndcg_5 = [], []
    for label, score in collections:
        group = [(l, s) for l, s in zip(label, score)]
        group = sorted(group, key=lambda x: x[1], reverse=True)
        group = [l for l, s in group]
        ndcg_3.append(NDCG(group, 3))
        ndcg_5.append(NDCG(group, 5))
    print(f'[!] NDCG@3: {round(np.mean(ndcg_3), 4)}')
    print(f'[!] NDCG@5: {round(np.mean(ndcg_5), 4)}')


def main_rerank_fg(**args):
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    if args['model'] in args['no_test_models']:
        print(f'[!] model {args["model"]} doesn"t support test')
        return
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)
    collections = agent.test_model_fg(test_iter, print_output=True)
    for name in collections:
        sbm_scores, weighttau_scores = [], []
        for label, score in collections[name]:
            r1 = [(i, j) for i, j in zip(range(len(label)), label)]
            r1 = [i for i, j in sorted(r1, key=lambda x: x[1], reverse=True)]
            r2 = [(i, j) for i, j in zip(range(len(score)), score)]
            r2 = [i for i, j in sorted(r2, key=lambda x: x[1], reverse=True)]
            sbm_scores.append(SBM(r1, r2))
            weighttau_scores.append(kendalltau_score(r1, r2))
        print(f'[!] Set Based Metric of Annotator {name}: {round(np.mean(sbm_scores), 4)}')
        print(f'[!] Weighted Kendall Tau of Annotator {name}: {round(np.mean(weighttau_scores), 4)}')


def main_rerank_time(**args):
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    if args['model'] in args['no_test_models']:
        print(f'[!] model {args["model"]} doesn"t support test')
        return
    
    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)

    bt = time.time()
    outputs = agent.test_model(test_iter, print_output=False, rerank_agent=None, core_time=True)
    cost_time = outputs['core_time']*1000
    cost_time /= len(test_iter)

    with open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_result_rerank_{pretrained_model_name}.txt', 'w') as f:
        for key, value in outputs.items():
            print(f'{key}: {value}', file=f)
        print(f'Cost-Time: {round(cost_time, 2)} ms', file=f)
    for key, value in outputs.items():
        print(f'{key}: {value}')

def main_generation(**args):
    args['mode'] = 'test'
    new_args = deepcopy(args)
    config = load_config(args)
    args.update(config)

    random.seed(args['seed'])
    torch.manual_seed(args['seed'])
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(args['seed'])

    test_data, test_iter, _ = load_dataset(args)
    agent = load_model(args)
    
    pretrained_model_name = args['pretrained_model'].replace('/', '_')
    save_path = f'{args["root_dir"]}/ckpt/{args["dataset"]}/{args["model"]}/best_{pretrained_model_name}.pt'
    agent.load_model(save_path)

    bt = time.time()
    outputs = agent.test_model(test_iter, print_output=False)
    cost_time = time.time() - bt
    cost_time *= 1000    # ms
    cost_time /= len(test_iter)

    with open(f'{args["root_dir"]}/rest/{args["dataset"]}/{args["model"]}/test_result_rerank_{pretrained_model_name}.txt', 'w') as f:
        for key, value in outputs.items():
            print(f'{key}: {value}', file=f)
        print(f'Cost-Time: {round(cost_time, 2)} ms', file=f)
    for key, value in outputs.items():
        print(f'{key}: {value}')
    print(f'Cost-Time: {round(cost_time, 2)} ms')


if __name__ == "__main__":
    args = vars(parser_args())
    if args['mode'] == 'recall':
        print(f'[!] Make sure that the inference script of model({args["model"]}) on dataset({args["dataset"]}) has been done.')
        main_recall(**args)
    elif args['mode'] == 'es_recall':
        main_es_recall(**args)
    elif args['mode'] == 'rerank':
        main_rerank(**args)
    elif args['mode'] == 'horse_human':
        main_horse_human(**args)
    elif args['mode'] == 'generation':
        main_generation(**args)
    elif args['mode'] == 'fg_rerank':
        main_rerank_fg(**args)
    elif args['mode'] == 'compare':
        main_compare(**args)
    elif args['mode'] == 'rerank_time':
        main_rerank_time(**args)
    else:
        raise Exception(f'[!] Unknown mode: {args["mode"]}')
