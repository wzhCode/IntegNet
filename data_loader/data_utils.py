
import numpy as np
import pandas as pd

from utils.math_utils import z_score


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.mean = stats['mean']
        self.std = stats['std']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'mean': self.mean, 'std': self.std}

    def get_len(self, type):
        return len(self.__data[type])


def seq_gen(len_seq, data_seq, offset, n_his, n_pred, n_dcpair, day_slot, c_in=1):
    '''
    Generate data sequence.
    return: [len_seq, n_frame, n_dcpair, c_in].
    '''
    n_frame = n_his + n_pred

    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_dcpair, c_in))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_dcpair, c_in])
    return tmp_seq


def read_file(file_path):
    '''
    Traffic data load function.
    '''
    data_seq = pd.read_csv(file_path, header=None).values
    print('V shape: ', data_seq.shape)
    data_seq = np.log(data_seq)  # log transform

    return data_seq


def data_gen(file_path, data_config, n_dcpair, n_his, n_pred, day_slot):
    '''
    Training, Validation, and Testing datasets generation.
    '''
    n_train, n_val, n_test = data_config

    data_seq = read_file(file_path)

    seq_train = seq_gen(n_train, data_seq, 0, n_his, n_pred, n_dcpair, day_slot)
    seq_val = seq_gen(n_val, data_seq, n_train, n_his, n_pred, n_dcpair, day_slot)
    seq_test = seq_gen(n_test, data_seq, n_train + n_val, n_his, n_pred, n_dcpair, day_slot)

    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)

    return dataset


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False):
    '''
    Data iterator in batch.
    '''
    len_inputs = len(inputs)

    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]
