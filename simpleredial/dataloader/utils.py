from header import *


def read_json_data_dual_bert(path, lang='zh'):
    with open(path) as f:
        dataset = []
        responses = []
        for line in tqdm(list(f.readlines())):
            line = line.strip()
            item = json.loads(line)
            context = item['q']
            response = item['r'].strip()
            dataset.append((1, context + [response]))
    print(f'[!] load {len(dataset)} samples from {path}')
    return dataset


def read_json_data_dual_bert_full(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in tqdm(list(f.readlines())):
            line = line.strip()
            item = json.loads(line)
            utterances = item['q'] + [item['r']]
            start_num = max(1, len(utterances) - 5)
            for i in range(start_num, len(utterances)):
                dataset.append((1, utterances[:i+1]))
    print(f'[!] load {len(dataset)} samples from {path}')
    return dataset


def read_json_data_arxiv(path, lang='zh'):
    with open(path) as f:
        responses_ = []
        for line in tqdm(list(f.readlines())):
            item = json.loads(line.strip())
            responses_.append(item['r'])
            responses_.extend(item['q'])

    with open(path) as f:
        dataset = []
        for line in tqdm(list(f.readlines())):
            line = line.strip()
            item = json.loads(line)
            context = item['q']
            response = item['r']
            if len(context) < 9:
                candidates = context + random.sample(responses_, 9 - len(context))
            else:
                candidates = random.sample(context, 9)
            dataset.append((context, response, candidates))
    print(f'[!] load {len(dataset)} samples from {path}')
    return dataset


def read_json_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        responses = []
        for line in tqdm(list(f.readlines())):
            line = line.strip()
            item = json.loads(line)
            context = item['q']
            response = item['r'].strip()
            # NOTE: the candidates may not be 10 (less than 10)
            candidates = [i.strip() for i in item['nr'] if i.strip()]
            dataset.append((context, response, candidates))
            responses.extend([response] + candidates)
        responses = list(set(responses))
    print(f'[!] load {len(dataset)} samples from {path}')
    print(f'[!] load {len(responses)} unique utterances in {path}')
    return dataset, responses


def read_text_data_utterances_compare_test(path, lang='zh', size=10):
    with open(path) as f:
        responses = []
        for line in tqdm(list(f.readlines())):
            line = line.strip().split('\t')[1:]
            responses.extend(line)
        responses_ = list(set(responses))

    with open(path) as f:
        dataset = []
        cache = []
        for line in tqdm(list(f.readlines())):
            if len(cache) == size:
                lines = [line.strip().split('\t') for line in cache]
                lines = [(line[0], line[1:]) for line in lines]     # (label, utterance list)

                context = lines[0][1][1:-1]
                responses = [line[1][-1] for line in lines if line[0] == '1']
                if len(responses) != 0:
                    response = random.choice(responses)    # random select one as the ground-truth
                    hard_negative_samples = [line[1][-1] for line in lines if line[0] == '0']
                    easy_negative_samples = random.sample(responses_, len(hard_negative_samples))
                    dataset.append((
                        context, 
                        response, 
                        hard_negative_samples, 
                        easy_negative_samples
                    ))
                else:
                    # ignore this case
                    pass
                cache = []
            cache.append(line)
    print(f'[!] load {len(dataset)} samples from {path}')
    return dataset


def read_json_data_from_gray_file(path):
    with open(path) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            item = json.loads(line.strip())
            context, response = item['q'], item['r']
            hn = item['nr']
            dataset.append((context, response, hn))
        print(f'[!] collect {len(dataset)} samples')
    return dataset


def read_text_data_utterances_compare(path, lang='zh'):
    '''read from the train_gray dataset''' 
    with open(path) as f:
        dataset = []
        for line in tqdm(list(f.readlines())):
            item = json.loads(line.strip())
            context = item['q']
            response = item['r']
            hard_negative_samples = item['nr']
            super_hard_negative_samples = item['snr']
            dataset.append((
                context, 
                response, 
                hard_negative_samples, 
                super_hard_negative_samples
            ))
    print(f'[!] load {len(dataset)} samples from {path}')
    return dataset


def read_text_data_utterances(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            dataset.append((label, utterances))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset

def read_text_data_with_neg_inner_session_neg(path, lang='zh'):
    with open(path) as f:
        dataset, responses = [], []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if label == 0:
                continue
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            context = utterances[:-1]
            response = utterances[-1]
            candidates = utterances[:-1]
            dataset.append((context, response, candidates))
            responses.extend(utterances)
    responses = list(set(responses))
    print(f'[!] load {len(dataset)} samples from {path}')
    print(f'[!] load {len(responses)} utterances from {path}')
    return dataset, responses


def read_text_data_with_super_hard_q_r(path, lang='zh'):
    path = f'{os.path.splitext(path)[0]}_gray.txt'
    with open(path) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            line = json.loads(line.strip())
            context = line['q']
            response = line['r']
            candidates = line['snr']
            dataset.append((context, response, candidates))
    print(f'[!] load {len(dataset)} samples from {path}')
    return dataset

def read_text_data_one2many(path, lang='zh'):
    path_ = f'{os.path.splitext(path)[0]}_gray.txt'
    with open(path_) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            line = json.loads(line.strip())
            q, r = line['q'], line['r']
            cands = line['snr']
            dataset.append((q, r, cands))
    print(f'[!] load {len(dataset)} samples from {path_}')
    return dataset

def read_text_data_one2many_replace(path, lang='zh'):
    path_ = f'{os.path.splitext(path)[0]}_gray.txt'
    with open(path_) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            line = json.loads(line.strip())
            q, r = line['q'], line['r']
            cands = line['hp']
            bad_response = line['bad_response']
            dataset.append((q, r, cands, bad_response))
    print(f'[!] load {len(dataset)} samples from {path_}')
    return dataset


def read_text_data_one2many_pesudo(path, lang='zh'):
    path_ = f'{os.path.splitext(path)[0]}_gray.txt'
    with open(path_) as f:
        dataset = []
        for line in f.readlines():
            line = json.loads(line.strip())
            context = line['q']
            response = line['r']
            candidates = line['snr']
            dataset.append((context, response, candidates))
    ext_path = f'{os.path.splitext(path)[0]}_gray_unparallel.txt'
    with open(ext_path) as f:
        for line in tqdm(f.readlines()):
            line = json.loads(line.strip())
            context = line['q']
            response = line['snr'][0]
            candidates = line['snr'][1:]
            dataset.append(([context], response, candidates))
    print(f'[!] load {len(dataset)} samples from {path} and {ext_path}')
    return dataset


def read_text_data_with_neg_q_r_neg(path, lang='zh'):
    path = f'{os.path.splitext(path)[0]}_gray.txt'
    with open(path) as f:
        dataset, responses = [], []
        for line in f.readlines():
            line = json.loads(line.strip())
            context = line['q']
            response = line['r']
            candidates = line['nr']
            dataset.append((context, response, candidates))
            responses.extend(context + [response] + candidates)
        responses = list(set(responses))
    print(f'[!] load {len(dataset)} samples from {path}')
    print(f'[!] load {len(responses)} utterances from {path}')
    return dataset, responses

def read_text_data_dual_bert(path, lang='zh'):
    sep = ' [SEP] '
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip().split('\t')
            label, utterances = int(line[0]), line[1:]
            if lang == 'zh':
                utterances = [''.join(u.split()) for u in utterances]
            context, response = sep.join(utterances[:-1]), utterances[-1]
            dataset.append((label, context, response))
    print(f'[!] load {len(dataset)} utterances from {path}')
    return dataset


def read_response_data_full(path, lang='zh', turn_length=5):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            utterance = line.strip().split('\t')
            if int(utterance[0]) == 0:
                dataset.append(utterance[-1])
                continue
            utterance = utterance[1:]
            if len(utterance) > turn_length:
                # only part of the utterances will be used for inference
                utterance = utterance[-turn_length:]
            if lang == 'zh':
                utterance = [''.join(i.split()) for i in utterance]
            dataset.extend(utterance)
    # delete the duplicate responses
    dataset = list(set(dataset))
    print(f'[!] load {len(dataset)} responses from {path}')
    return dataset

def read_response_and_context_full(path, lang='zh', turn_length=5):
    dataset = read_text_data_utterances(path, lang=lang)
    context, response = [], []
    for label, utterances in dataset:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - turn_length)
        for i in range(start_num, len(utterances)):
            # i is the index of the response
            context.append(utterances[:i])
            response.append(utterances[i])
    print(f'[!] collect {len(context)} samples for training')
    return context, response

def read_response_data_with_context(path, lang='zh'):
    with open(path) as f:
        dataset = {}
        for line in f.readlines():
            utterance = line.strip().split('\t')
            if int(utterance[0]) == 0:
               continue
            ctx, res = utterance[1:-1], utterance[-1]
            if res in dataset:
                dataset[res].append(ctx)
            else:
                dataset[res] = [ctx]
        new_dataset = {}
        for i, j in dataset.items():
            if len(j) == 1:
                new_dataset[i] = j
    print(f'[!] load {len(new_dataset)} responses from {path}')
    return new_dataset


def read_response_data_test(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            utterance = line.strip().split('\t')
            utterance = utterance[-1]
            if lang == 'zh':
                utterance = ''.join(utterance.split())
            dataset.append(utterance)
    # delete the duplicate responses
    dataset = list(set(dataset))
    print(f'[!] load {len(dataset)} test responses from {path}')
    return dataset


def read_response_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            utterance = line.strip().split('\t')
            if int(utterance[0]) == 0:
               continue
            utterance = utterance[-1]
            if lang == 'zh':
                utterance = ''.join(utterance.split())
            dataset.append(utterance)
    # delete the duplicate responses
    dataset = list(set(dataset))
    print(f'[!] load {len(dataset)} responses from {path}')
    return dataset


def read_cl_response_data(path, lang='zh', max_context_turn=0):
    with open(path) as f:
        dataset = {}
        for line in f.readlines():
            utterances = line.strip().split('\t')
            if int(utterances[0]) == 0:
                continue
            utterances = utterances[1:]
            if lang == 'zh':
                utterances = [''.join(i.split()) for i in utterances]

            if max_context_turn == 0:
                context = ' [SEP] '.join(utterances[:-1])
            else:
                context = ' [SEP] '.join(utterances[-max_context_turn-1:-1])

            response = utterances[-1]
            if response in dataset:
                dataset[response].append(context)
            else:
                dataset[response] = [context]
        print(f'[!] load {len(dataset)} responses from {path}')
        return dataset
            

def read_response_json_data(path, lang='zh'):
    with open(path) as f:
        dataset = []
        pbar = tqdm(f.readlines())
        for line in pbar: 
            dataset.append(line.strip())
        print(f'[!] already collect {len(dataset)} utterances for inference')
    return dataset


def read_text_data_with_source(path, lang='zh'):
    with open(path) as f:
        dataset = []
        for line in tqdm(f.readlines()):
            line = line.strip().split('\t')
            context = line[0]
            response = json.loads(line[1])
            dataset.append(([context], response[0], response[1], response[2]))
        print(f'[!] collect: {len(dataset)} utterances for inference')
    return dataset

def read_text_data_from_log_file(path, lang='zh'):
    '''
    [Context ] context sentence ....
    [Resoibse] response setnecen ....
    
    [Context ] context sentence2 ....
    [Response] response sentencen2 ....'''
    with open(path) as f:
        dataset = []
        lines = f.readlines()
        assert len(lines) % 3 == 0

        for i in range(0, len(lines), 3):
            session = lines[i:i+3]
            context = session[0].lstrip('[Context ]').strip()
            response = session[1].lstrip('[Response]').strip()
            dataset.append((context, response))
        print(f'[!] collect {len(dataset)} sessions')
    return dataset

def read_essay_dataset(path):
    with open(path) as f:
        dataset = []    # essay
        responses = []
        for line in f.readlines():
            line = line.strip().split('\t')
            essay_id, passage_id, _, sentence_id, _, sentence = line
            essay_id = int(essay_id)
            passage_id = int(passage_id)
            sentence_id = int(sentence_id)
            sentence = sentence.replace('|', '')
            if len(sentence) < 3:
                continue
            dataset.append((
                essay_id, passage_id, sentence_id, sentence    
            ))
            responses.append(sentence)
        responses = list(set(responses))

        data, cache = []
        last_p_id = -1
        for e_id, p_id, _, s in dataset:
            if p_id != last_p_id:
                if cache:
                    data.append(cache)
                cache = [s]
                if last_p_id != p_id:
                    last_p_id = p_id
            else:
                cache.append(s)
        if cache:
            data.append(cache)
        print(f'[!] collect {len(data)} sentences')
    return dataset, responses

def read_text_data_utterances_and_pesudo_pairs(path1, path2, lang='zh'):
    dataset1 = read_text_data_utterances(path1, lang=lang)
    with open(path2) as f:
        dataset2 = []
        for line in f.readlines():
            line = json.loads(line.strip())
            # pos = random.choice(line['snr'])
            # utterances = [line['q'], pos]
            utterances = [line['q'], line['snr'][0]]
            dataset2.append((1, utterances))
    return dataset1 + dataset2

def read_text_data_utterances_full_da_ctx(path, lang='zh', turn_length=5, da_ctx_num=1):
    '''the full conversation context will be used'''
    dataset = read_text_data_utterances(path, lang=lang)
    data = []
    aug_counter = 0
    for label, utterances in dataset:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - turn_length)
        for i in range(start_num, len(utterances)):
            # i is the index of the response
            data.append((1, utterances[:i+1]))
            current_ctx_length = len(utterances[:i])
            da_ctx_num_ = min(current_ctx_length-1, da_ctx_num)
            if da_ctx_num_ == 0:
                continue
            for ii in range(da_ctx_num_):
                ctx_length = current_ctx_length - ii - 1
                sample = utterances[i-ctx_length:i+1]
                data.append((1, sample))
                aug_counter += 1
    print(f'[!] collect {len(data)} samples for training; augmentation size: {aug_counter}')
    return data


def read_text_data_utterances_full_large(path, lang='zh', turn_length=5, min_length=3):
    '''the full conversation context will be used, min_length means the min turn length of the conversation (include the response)'''
    dataset = read_text_data_utterances(path, lang=lang)
    data = []
    for label, utterances in dataset:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - turn_length)
        for idx in range(start_num, len(utterances)):
            data.append((1, utterances[:idx+1]))
            for j in range(1, idx):
                if len(utterances[j:idx+1]) >= min_length:
                    data.append((1, utterances[j:idx+1]))
    print(f'[!] collect {len(data)} samples for training')
    return data


def read_text_data_utterances_full_data_filter(path):
    '''the full conversation context will be used'''
    dataset = torch.load(path)
    data = []
    for ctx, res, _ in dataset:
        data.append((1, ctx + [res]))
    print(f'[!] collect {len(data)} samples for training')
    return data


def read_text_data_utterances_full_ft(path, lang='zh', turn_length=5):
    '''the full conversation context will be used'''
    dataset = read_text_data_utterances(path, lang=lang)
    responses = [us[-1] for _, us in dataset]
    responses = list(set(responses))
    data = []
    for label, utterances in dataset:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - turn_length)
        for i in range(start_num, len(utterances)):
            # i is the index of the response
            # positive
            data.append((1, utterances[:i+1]))
            # negative
            data.append((0, utterances[:i] + [random.choice(responses)]))
    print(f'[!] collect {len(data)} samples for training')
    return data


def read_text_data_utterances_full(path, lang='zh', turn_length=5):
    '''the full conversation context will be used'''
    dataset = read_text_data_utterances(path, lang=lang)
    data = []
    for label, utterances in dataset:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - turn_length)
        for i in range(start_num, len(utterances)):
            # i is the index of the response
            data.append((1, utterances[:i+1]))
    print(f'[!] collect {len(data)} samples for training')
    return data

def read_text_data_utterances_full_fake_ctx(path1, path2, lang='zh'):
    dataset = read_text_data_utterances(path1, lang=lang)
    data = []
    for label, utterances in dataset:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - 5)
        for i in range(start_num, len(utterances)):
            data.append((1, utterances[:i+1]))
    ext_data = []
    with open(path2) as f:
        for line in tqdm(f.readlines()):
            line = line.strip()
            ext_data.append((1, [line, line]))
    print(f'[!] collect {len(data)} pair samples and {len(ext_data)} monolingual samples')
    return data, ext_data

def read_text_data_utterances_and_full_and_pesudo_pairs(path1, path2, lang='zh', turn_length=5):
    dataset1_ = read_text_data_utterances(path1, lang=lang)
    dataset1 = []
    for label, utterances in dataset1_:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - turn_length)
        for i in range(start_num, len(utterances)):
            dataset1.append((1, utterances[:i+1]))
    with open(path2) as f:
        dataset2 = []
        for line in tqdm(f.readlines()):
            line = json.loads(line.strip())
            utterances = [line['q'], line['snr'][0]]
            dataset2.append((1, utterances))
    dataset = dataset1 + dataset2
    print(f'[!] collect {len(dataset)} training samples')
    return dataset

def read_extended_douban_corpus(path):
    with open(path) as f:
        dataset = []
        for line in f.readlines():
            line = line.strip()
            dataset.append(line)
    print(f'[!] read {len(dataset)} utterances from extended douban corpus')
    return dataset


def read_text_data_utterances_and_full_and_pesudo_pairs_ft(path1, path2, lang='zh'):
    dataset1_ = read_text_data_utterances(path1, lang=lang)

    # negative utterances pool
    utterances = []
    for label, us in dataset1_:
        utterances.extend(us)
    with open(path2) as f:
        dataset2 = []
        for line in f.readlines():
            line = json.loads(line.strip())
            utterances.extend(line['snr'])
    utterances_pool = list(set(utterances))

    dataset1 = []
    for label, utterances in dataset1_:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - 5)
        for i in range(start_num, len(utterances)):
            dataset1.append((1, utterances[:i+1]))
            neg = random.choice(utterances_pool)
            dataset1.append((0, utterances[:i] + [neg]))
    with open(path2) as f:
        dataset2 = []
        for line in f.readlines():
            line = json.loads(line.strip())
            # pos = random.choice(line['snr'])
            # utterances = [line['q'], pos]
            utterances = [line['q'], line['snr'][0]]
            dataset2.append((1, utterances))
            neg = random.choice(utterances_pool)
            dataset2.append((0, [line['q'], neg]))
    return dataset1 + dataset2


def read_text_data_utterances_full_neg_session(path, lang='zh'):
    '''the full conversation context will be used; the utterance in the same context will be treated as the hard negative samples'''
    dataset = read_text_data_utterances(path, lang=lang)
    data = []
    for label, utterances in dataset:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - 5)
        for i in range(start_num, len(utterances)):
            # neg_session samples
            neg_session = deepcopy(utterances)
            neg_session.remove(utterances[i])
            data.append((1, utterances[:i+1], neg_session))
    print(f'[!] collect {len(data)} samples for training')
    return data


def read_text_data_ext_utterances_full(path, lang='zh', turn_length=5):
    '''the full conversation context will be used'''
    dataset = read_text_data_utterances(path, lang=lang)
    data = []
    for label, utterances in dataset:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - turn_length)
        for i in range(start_num, len(utterances)):
            # i is the index of the response
            data.append((1, utterances[:i+1]))
    print(f'[!] collect {len(data)} samples from the extended corpus: {path}')
    return data

def read_text_data_line_by_line(path):
    dataset = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            line = ''.join(line.strip().split())
            dataset.append(line)
    print(f'[!] read {len(dataset)} lines')
    return dataset

def read_text_data_unlikelyhood(path):
    dataset = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            line = ''.join(line.strip().split())
            lines = [u.strip() for u in re.split('(，|。|；|！|？)', line) if u.strip()]
            items = []
            for item in lines:
                if item in ['，', '。', '；', '！', '？']:
                    if len(items) > 0:
                        items[-1] += item
                    else:
                        items.append(item)
                else:
                    items.append(item)
            dataset.append(items)
    return dataset

def read_text_data_unlikelyhood_test(path):
    dataset = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            dataset.append(line.strip())
    return dataset


def read_torch_data_bert_mask(path, hard_negative_num=5):
    contexts, responses, candidates = torch.load(path)
    pool = list(set(responses))
    dataset = []
    for c, r, cand in tqdm(list(zip(contexts, responses, candidates))):
        dataset.append((c, r, cand))
    print(f'[!] collect {len(dataset)} training samples from bert mask inference results')
    return dataset

def read_torch_data_simcse(path, lang='zh'):
    dataset = torch.load(path)
    data, data_cands = [], []
    for _, value in dataset.items():
        utterances = [i['text'] for i in value]
        candidates = [i['cands'] for i in value]
        data.append(utterances)
        data_cands.append(candidates)
    print(f'[!] collect {len(data)} samples for training')
    return data, data_cands


def read_bm25_hard_negative(path):
    dataset = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            item = json.loads(line.strip())
            dataset.append(item)
    print(f'[!] load {len(dataset)} training samples')
    return dataset

def read_dpr_gray(path):
    dataset = []
    with open(path) as f:
        for line in tqdm(f.readlines()):
            dataset.append(json.loads(line))
    print(f'[!] collect {len(dataset)} samples')
    return dataset


def read_text_data_utterances_full_add_next_history_prediction(path, lang='zh', turn_length=5):
    '''the full conversation context will be used'''
    dataset = read_text_data_utterances(path, lang=lang)
    data = []
    for label, utterances in dataset:
        if label == 0:
            continue
        start_num = max(1, len(utterances) - turn_length)
        for i in range(start_num, len(utterances)):
            # format: [label, contexts, responses]
            data.append((1, utterances[:i], utterances[i:]))
    print(f'[!] collect {len(data)} samples for training')
    return data
