import numpy as np
import torch
from sklearn.metrics import f1_score
try:
    import ipdb
except:
    pass

# POI Metrics
def p_rec(tops, labels, k):
    """
    Calculate the recall score for top-k recommendations.
    Returns:
        float: Recall
    This function computes the recall, which measures the proportion of actual items of interest that are
    included in the top-k recommendations. The function iterates over pairs of top-k recommendations and
    corresponding actual items, calculates the recall for each user, and then averages these values.
    """
    res = 0.
    for _, (top, label) in enumerate(zip(tops, labels)):
        hit = np.intersect1d(top[:k], label) # 计算它们的交集
        r = len(hit) / (len(set(label)) - 1)
        res += r
    return res

def p_precision(tops, labels, k):
    """
    Calculate the precision score for top-k recommendations.
    Returns:
        float
    This function computes the precision, which measures the proportion of recommended items in the top-k list
    that are actual items of interest. The function iterates over pairs of top-k recommendations and corresponding
    actual items, calculates the precision for each user, and then averages these values.
    """
    res = 0.
    for _, (top, label) in enumerate(zip(tops, labels)):
        hit = np.intersect1d(top[:k], label)
        r = len(hit) / k # 1 for '0'
        res += r
    return res

def p_f1(tops, labels, k):
    """
    Calculate the F1 score for top-k recommendations.
    Returns:
        float
    This function computes the F1 score, which is the harmonic mean of precision and recall. The function iterates
    over pairs of top-k recommendations and corresponding actual items, calculates the F1 score for each user,
    and then averages these values. It handles cases where the denominator in the F1 score calculation is zero.
    """
    res = 0.
    for _, (top, label) in enumerate(zip(tops, labels)):
        hit = np.intersect1d(top[:k], label)
        p = len(hit) / k # 1 for '0'
        r = len(hit) / (len(set(label)) - 1) # 1 for '0'
        try:
            res += (2 * p * r / (p + r))
        except:
            res += 0
    return res

def p_ndcg(tops, labels, k):
    """
    Calculate the normalized discounted cumulative gain (NDCG) for top-k recommendations.
    Returns:
        float
    This function computes the NDCG, a measure of ranking quality. For each user, it calculates the DCG (Discounted
    Cumulative Gain) and IDCG (Ideal DCG) and then normalizes the DCG by IDCG to get the NDCG score. It averages
    these scores across all users. The function handles both relevant (rel = 1) and non-relevant (rel = 0) items.
    """
    res = 0.
    for top, label in zip(tops, labels):
        dcg = 0.
        idcg = 0.
        for i, p in enumerate(top[:k], start=1):
            rel = 1 if np.isin(p, label) else 0
            dcg += (2 ** rel - 1) / (np.log2(i + 1))
            idcg += 1 / (np.log2(i + 1))
        ndcg = dcg / idcg
        res += ndcg
    return res

# Region Metrics
def r_map(tops, labels, weight=None):
    """
    Calculate the mean average precision (MAP) for region-based recommendations.
    Returns:
        float
    This function computes MAP, a measure that considers the order of recommendations. It iterates over pairs
    of recommended and actual regions, calculates the precision at each relevant item found, averages these
    precision values, and then averages across all instances. If weights are provided, they are applied to
    each instance's precision.
    """
    map_ = []
    for instance_idx, (top, label) in enumerate(zip(tops, labels)):
        m = 0.
        relative_num = 0.
        for i, k in enumerate(top,start=1):
            if k == label:
                m += (relative_num + 1) / i
                relative_num += 1
        if relative_num > 0:
            m /= relative_num
        if weight: m *= weight[instance_idx]
        map_.append(m)
    return np.mean(map_)

def r_precision(tops, labels, weight=None):
    """
    Calculate the precision for region-based recommendations.
    Returns:
        float
    This function computes precision for region-based recommendations. It iterates over each pair of recommended
    and actual regions, calculates the proportion of correctly predicted regions, and then averages these values
    across all instances. If weights are provided, they are applied to each instance's precision score.
    """
    res = []
    tops = tops.cpu().detach().numpy()
    labels = labels.cpu().detach().numpy()
    for instance_idx, (ps, l) in enumerate(zip(tops, labels)):
        # ipdb.set_trace()
        showup = np.sum(ps == l)
        prec = showup / len(ps)
        if weight: prec *= weight[instance_idx]
        res.append(prec)
    return np.mean(res)

def r_acc(predict, label):
    """
    Calculate the accuracy for region-based recommendations.
    Returns:
        float
    This function computes accuracy, which is the ratio of correctly predicted regions to the total number
    of predictions. It compares the predicted regions with the actual regions of interest and calculates the
    proportion of correct predictions.
    """
    return torch.sum(predict == label) / label.size(0)

def r_f1(predict, label, avg):
    """
    Calculate the F1 score for region-based recommendations.
    Returns:
        float
    The F1 score is a measure of a test's accuracy and considers both the precision and recall. It is the
    harmonic mean of precision and recall. This function computes the F1 score for the given predictions and
    labels, applying the specified averaging method.
    """
    return f1_score(label.cpu(), predict.cpu(), average=avg)

def weight_func(x):
    """
    Calculate a weighting factor based on the input value.
    Returns:
        float
    This function calculates a weighting factor using a cosine function. It is often used to transform a value
    (like similarity or relevance scores) into a weighting factor that can be used in further calculations or
    algorithms. The transformation is designed to decrease the weight as the input value increases.
    """
    return np.cos(np.pi / 2 * x * 10)

# ============================= Trip metrics =========================== #
def f1_score(target, predict, noloop=False):
    """
    Compute F1 Score for recommended trajectories
    :param target: the actual trajectory
    :param predict: the predict trajectory
    :param noloop:

    :return: f1
    """
    assert (isinstance(noloop, bool))
    assert (len(target) > 0)
    assert (len(predict) > 0)

    if noloop:
        intersize = len(set(target) & set(predict))
    else:
        match_tags = np.zeros(len(target), dtype=np.bool_)
        for poi in predict:
            for j in range(len(target)):
                if not match_tags[j] and poi == target[j]:
                    match_tags[j] = True
                    break
        intersize = np.nonzero(match_tags)[0].shape[0]

    recall = intersize * 1.0 / len(target)
    precision = intersize * 1.0 / len(predict)
    denominator = recall + precision
    if denominator == 0:
        denominator = 1

    f1 = 2 * precision * recall * 1.0 / denominator

    return f1


def pairs_f1_score(target, predict):
    """
    Compute Pairs_F1 Score for recommended trajectories
    :param target:
    :param predict:
    :return: pairs_f1
    """
    # Check if number of elements > 0
    assert target.numel() > 0
    n = target.numel()
    nr = predict.numel()
    if n == 1 or nr == 1:
        return 1.0 if target.item() == predict.item() else 0.0
    n0 = n * (n - 1) / 2
    n0r = nr * (nr - 1) / 2

    order_dict = dict()
    for i, poi in enumerate(target):
        order_dict[poi.item()] = i

    nc = 0
    for i in range(nr):
        poi1 = predict[i].item()
        for j in range(i + 1, nr):
            poi2 = predict[j].item()
            if poi1 in order_dict and poi2 in order_dict and poi1 != poi2:
                if order_dict[poi1] < order_dict[poi2]:
                    nc += 1

    precision = (1.0 * nc) / (1.0 * n0r)
    recall = (1.0 * nc) / (1.0 * n0)
    if nc == 0:
        pairs_f1 = 0
    else:
        pairs_f1 = 2. * precision * recall / (precision + recall)

    return pairs_f1

def count_repetition_percentage(input_data):
    # if list
    if isinstance(input_data, list):
        unique_items = set(input_data)
    # if tensor
    elif hasattr(input_data, 'numpy'):
        unique_items = set(input_data.cpu().numpy().tolist())
    else:
        raise ValueError("Input data must be a list or a tensor.")

    total_items = len(input_data)
    repetition_items_count = total_items - len(unique_items)
    repetition_ratio = repetition_items_count / total_items

    return repetition_ratio


def count_adjacent_repetition_rate(input_data):
    if isinstance(input_data, list):
        predictions = input_data
    elif hasattr(input_data, 'numpy'):
        predictions = input_data.cpu().numpy().flatten().tolist()
    else:
        raise ValueError("Input data must be a list or a tensor.")

    total = len(predictions)
    if total < 2:
        return 0.0

    repeated = sum(1 for i in range(1, total) if predictions[i] == predictions[i - 1])
    repetition_ratio = repeated / (total - 1)

    return repetition_ratio