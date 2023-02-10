# LSTM model

import tensorflow as tf
import numpy as np
import csv
import time
import pandas as pd

from tensorflow.keras.layers import Activation, Dropout, BatchNormalization, Activation
from tensorflow.keras.layers import LSTM, TimeDistributed, Flatten, Dense
from tensorflow.keras.models import load_model
#from tensorflow.keras.callbacks import LearningRateScheduler

from models.integnet_model import my_cost_loss
from utils.math_utils import z_score, z_inverse, Cost, headroom, evaluation
from models.tester import save_results



def seq_gen_lstm(len_seq, data_seq, offset, n_his, n_pred, n_dcpair, day_slot):
    n_frame = n_his + n_pred

    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_dcpair, n_frame))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :] = np.reshape(data_seq[sta:end, :].T, [n_dcpair, n_frame])

    return tmp_seq  # (time_len, n_dcpair, n_frame)


def read_data_lstm(data_file, data_config, n_dcpair, n_his, n_pred, day_slot):
    n_train, n_val, n_test = data_config

    data_seq = pd.read_csv(data_file, header=None).values
    print('V shape: ', data_seq.shape)

    data_seq = np.log(data_seq)

    # seq_train, seq_val, seq_test: np.array, [time_len, n_dcpair, n_frame].
    seq_train = seq_gen_lstm(n_train, data_seq, 0, n_his, n_pred, n_dcpair, day_slot)
    seq_val = seq_gen_lstm(n_val, data_seq, n_train, n_his, n_pred, n_dcpair, day_slot)
    seq_test = seq_gen_lstm(n_test, data_seq, n_train + n_val, n_his, n_pred, n_dcpair, day_slot)

    x_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    x_train = z_score(seq_train, x_stats['mean'], x_stats['std'])
    x_val = z_score(seq_val, x_stats['mean'], x_stats['std'])
    x_test = z_score(seq_test, x_stats['mean'], x_stats['std'])
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}

    return [x_data, x_stats]


def model_train_lstm(x_data, n_his, n_pred, epoch, batch_size, learning_rate, model_file):
    train_data = x_data['train']

    train_X = train_data[:, :, 0:n_his]
    train_Y = train_data[:, :, n_his:n_his+n_pred]

    # train_X = np.transpose(train_X, (0, 2, 1))
    # train_Y = np.transpose(train_Y, (0, 2, 1))
    print('train data shape: ', train_X.shape, train_Y.shape)
    
    model = tf.keras.Sequential()
    model.add(LSTM(128, input_shape=(train_X.shape[1], train_X.shape[2]), return_sequences=True))
    model.add(Activation('relu'))
    model.add(TimeDistributed(Dense(1)))
    model.summary()
    
    valid_data = x_data['val']
    valid_X = valid_data[:, :, 0:n_his]
    valid_Y = valid_data[:, :, n_his:n_his + n_pred]
    
    # valid_X = np.transpose(valid_X, (0, 2, 1))
    # valid_Y = np.transpose(valid_Y, (0, 2, 1))
    
    model.compile(loss=my_cost_loss, optimizer='rmsprop')

    model.fit(train_X, train_Y, validation_data=(valid_X, valid_Y), epochs=epoch, batch_size=batch_size, verbose=2)
    model.save(model_file)
    print('Training model finished!')


def model_test_lstm(x_data, x_stats, n_his, n_pred, model_file, results_file):
    test_data = x_data['test']
    data_X = test_data[:, :, 0:n_his]
    data_Y = test_data[:, :, n_his:n_his + n_pred]

    # data_X = np.transpose(data_X, (0, 2, 1))
    # data_Y = np.transpose(data_Y, (0, 2, 1))

    data_seq = np.copy(data_X)

    model = load_model(model_file, custom_objects={'my_cost_loss': my_cost_loss})

    len_seq, n, _ = data_X.shape
    predict_Y = np.zeros((len_seq, n, n_pred))

    pred = model.predict(data_seq[:, :, 0:n_his])
    predict_Y[:, :, j] = pred[:, :, 0]

    [evl_test, results] = evaluation(data_Y, predict_Y, x_stats)

    save_results(results, results_file)

    te = evl_test
    print(f'Evaluation: Overcost {te[0]:6.3f}; Undercost {te[1]:6.3f}; Cost {te[2]:6.3f}. ')
    print('Testing model finished!')


if __name__ == '__main__':
    # parameter configuration
    n_dcpair = 330
    n_his = 12
    n_pred = 1
    data_config = [14, 2, 2]
    day_slot = 288

    epoch = 100
    batch_size = 32
    learning_rate = 1e-3

    data_file = 'file_path_traffic_data'  # your file path of traffic data
    model_file = 'file_path_save_model'  # save trained model
    results_file = 'file_path_results'  # save predict results

    [x_data, x_stats] = read_data_lstm(data_file, data_config, n_dcpair, n_his, n_pred, day_slot)

    print('Training model:')
    start = time.time()
    model_train_lstm(x_data, n_his, n_pred, epoch, batch_size, learning_rate, model_file)
    end = time.time()
    run_time = end - start
    print('Training time: ', run_time)
    print('Inference:')
    start = time.time()
    model_test_lstm(x_data, x_stats, n_his, n_pred, model_file, results_file)
    end = time.time()
    run_time = end - start
    print('Inference time: ', run_time)