from header import *
from scipy.stats import kendalltau, weightedtau


def SBM(r1, r2):
    '''Set based metrics'''
    assert len(r1) == len(r2)
    length = len(r1)
    scores = []
    set1, set2 = set(), set()
    for i in range(1, length):
        a, b = set(r1[:i]), set(r2[:i])
        num = len(a & b)
        scores.append(num / i)
    score = np.mean(scores)
    return score

def spearman_footrule(r1, r2):
    scores = []
    for idx, i in enumerate(r1):
        index = r2.index(i)
        scores.append(abs(idx - index))
    return sum(scores)

def kendalltau_score(r1, r2):
    score, _ = weightedtau(np.array(r1), np.array(r2), weigher=lambda x: x)
    return score

# NDCG
def DCG(label_list):
    dcgsum = 0
    for i in range(len(label_list)):
        dcg = (2**label_list[i] - 1)/math.log(i+2, 2)
        dcgsum += dcg
    return dcgsum

def NDCG(label_list, topk):
    '''NDCG@k'''
    dcg = DCG(label_list[:topk])
    ideal_list = sorted(label_list, reverse=True)
    ideal_dcg = DCG(ideal_list[:topk])
    if ideal_dcg == 0:
        return 0
    return dcg/ideal_dcg


if __name__ == "__main__":
    list1 = [0, 1, 2, 3, 4, 5]
    list2 = [1, 0, 2, 3, 4, 5]
    print(SBM(list1, list2))
    print(spearman_footrule(list1, list2))
    print(kendalltau_score(list1, list2))
    
    list1 = [0, 1, 2, 3, 4, 5]
    list2 = [0, 1, 2, 3, 5, 4]
    print(SBM(list1, list2))
    print(spearman_footrule(list1, list2))
    print(kendalltau_score(list1, list2))
