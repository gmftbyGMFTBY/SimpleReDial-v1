import random
import jieba
import ipdb
from itertools import combinations, chain
from random import shuffle
import re
from copy import deepcopy


def load_stop_words(path):
    stop_words = []
    with open(path) as f:
        for i in f.readlines():
            if i.strip():
                stop_words.append(i.strip())
    stop_words.append('')
    return set(stop_words)


########################################################################
# Sentence Swap
# randomly swap two sentecens in the multi-turn conversation context
########################################################################

def sentence_swap(sentences, alpha_ss=1):
    # alpha_ss: swap times, donot swap the last query
    if len(sentences) <= 2:
        return
    b, e = 0, len(sentences)-1
    for _ in range(alpha_ss):
        b_, e_ = random.randint(b, e-1), random.randint(b, e-1)
        if b_ == e_:
            continue
        sentences[b_], sentences[e_] = sentences[e_], sentences[b_]

########################################################################
# random replacement
# Replace n words in the sentence with random sampled word
########################################################################

def random_replacement(words, p, vocab):
    num_replaced = 0
    replace_idx_ = [[(i, j) for j, w in enumerate(subwords)] for i, subwords in enumerate(words)]
    replace_idx_ = list(chain(*replace_idx_))
    n = max(1, int(p * len(replace_idx_)))
    replace_idx = random.sample(replace_idx_, n)
    replace_idx_ = list(set(replace_idx_) - set(replace_idx))

    for i, j in replace_idx:
        random_word = random.choice(vocab)
        words[i][j] = random_word
        

########################################################################
# Random deletion
# Randomly delete words from the sentence with probability p
########################################################################

def random_deletion(words, p):
    delete_idx = [[(i, j) for j, w in enumerate(subwords)] for i, subwords in enumerate(words)]
    delete_idx = list(chain(*delete_idx))
    n = max(1, int(p * len(delete_idx)))
    delete_idx = random.sample(delete_idx, n)
    new_words = []
    for i, subwords in enumerate(words):
        cache = []
        for j, w in enumerate(subwords):
            if (i, j) not in delete_idx:
                cache.append(w)
        new_words.append(cache)
    return new_words

########################################################################
# Random swap
# Randomly swap two words in sentences n times
########################################################################

def random_swap(words, p):
    swap_idx = [[(i, j) for j, w in enumerate(subwords)] for i, subwords in enumerate(words)]
    swap_idx = list(chain(*swap_idx))
    n = max(1, int(p * len(swap_idx)))
    if len(swap_idx) < 2:
        return
    swap_idx = random.sample(list(combinations(swap_idx, 2)), n)
    for (a, b), (c, d) in swap_idx:
        words[a][b], words[c][d] = words[c][d], words[a][b]

########################################################################
# Random insertion
# Randomly insert n words into the sentence
########################################################################

def random_insertion(words, p, vocab):
    insert_idx_ = [[(i, j) for j, w in enumerate(subwords)] for i, subwords in enumerate(words)]
    insert_idx_ = list(chain(*insert_idx_))
    n = max(1, int(p * len(insert_idx_)))
    insert_idx = random.sample(insert_idx_, n)
    for i, j in insert_idx:
        random_word = random.choice(vocab)
        words[i].insert(j, random_word)

########################################################################
# main data augmentation function
########################################################################

def da(sentences, alpha_ss=2, alpha_sr=0.1, alpha_ri=0.1, alpha_rs=0.1, p_rd=0.1, num_aug=5, stop_words=None, lang='zh', vocab=None):

    # augmentation pipeline
    # 1. sentence swap
    # 2. random replacement
    # 3. random insertion
    # 4. random swap 
    # 5. random deletion
    augmented_sentences = []
    for _ in range(num_aug):
        sentences_ = deepcopy(sentences)
        # sentence swap
        sentence_swap(sentences_, alpha_ss=alpha_ss)

        ## prepare for the following augmentation
        if lang == 'zh':
            words = [list(jieba.cut(s)) for s in sentences_]
        else:
            words = [s.split(' ') for s in sentences_]

        counter = 0
        while True:
            try:
                ratio = random.random()
                if ratio >= 0.9:
                    # random replacement
                    random_replacement(words, alpha_sr, vocab)
                elif 0.8 <= ratio < 0.9:
                    # random insert
                    random_insertion(words, alpha_ri, vocab)
                elif 0.3 <= ratio < 0.8:
                    # random swap
                    random_swap(words, alpha_rs)
                else:
                    # random deletion
                    words = random_deletion(words, p_rd)
                counter += 1
                if counter >= 1:
                    break
            except:
                pass

        strings = [''.join(subwords) if lang == 'zh' else ' '.join(subwords) for subwords in words]
        augmented_sentences.append(strings)
    return augmented_sentences
