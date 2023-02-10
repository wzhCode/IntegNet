
import numpy as np
import math


def z_score(x, mean, std):
    '''
    Z-score normalization function.
    '''
    return (x - mean) / std


def z_inverse(x, mean, std):
    '''
    The inverse of function z_score().
    '''
    return x * std + mean


def Cost(v, v_):
    '''
    Evaluation metrics.
    '''
    diff = v_ - v
    cost = np.zeros(np.shape(diff))
    cost = np.where(diff >= 0, diff, cost)
    cost = np.where(diff < 0, 1 * np.abs(diff), cost)

    overcost = np.zeros(np.shape(diff))
    undercost = np.zeros(np.shape(diff))
    overcost = np.where(diff >= 0, diff, overcost)
    undercost = np.where(diff < 0, np.abs(diff), undercost)

    return [np.mean(np.sum(overcost, axis=1)), np.mean(np.sum(undercost, axis=1)), np.mean(np.sum(cost, axis=1))]


def headroom(v, v_):
    '''
    Setting aside headroom.
    '''
    head_v_ = np.zeros(v_.shape)
    for i in range(v.shape[1]):
        obs = v[:, i, 0]
        pre = v_[:, i, 0]
        diff = obs - pre
        mean_error = np.mean(diff)
        std_error = np.std(diff)
        head_v_[:, i, 0] = pre + mean_error + 1.96*std_error
    return head_v_


def evaluation(y, y_, x_stats):
    '''
    Evaluation function: calculate Overcost and Undercost between ground truth and prediction.
    '''
    # zscore inverse
    v = z_inverse(y, x_stats['mean'], x_stats['std'])
    v_ = z_inverse(y_, x_stats['mean'], x_stats['std'])

    # log inverse
    v = np.power(math.e, v)
    v_ = np.power(math.e, v_)

    # loss function + headroom
    head_v_ = headroom(v, v_)
    return [np.array([Cost(v, head_v_)[0], Cost(v, head_v_)[1], Cost(v, head_v_)[2]]), np.array([v, head_v_])]

