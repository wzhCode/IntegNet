# GAT model

import numpy as np
import csv
import pandas as pd
import time
import math
import tensorflow as tf
from tensorflow.keras.callbacks import EarlyStopping, CSVLogger
from tensorflow.keras.layers import Input, Dropout, Dense, Flatten, Concatenate
from tensorflow.keras.models import Model, load_model
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.regularizers import l2
from spektral.layers import GATConv
from spektral.transforms import LayerPreprocess, AdjToSpTensor

from models.integnet_model import my_cost_loss
from utils.math_utils import z_score, z_inverse, Cost, headroom, evaluation
from models.tester import save_results


def seq_gen_gat(len_seq, data_seq, offset, n_his, n_pred, n_dcpair, day_slot):
    n_frame = n_his + n_pred

    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_dcpair, n_frame))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :] = np.reshape(data_seq[sta:end, :].T, [n_dcpair, n_frame])

    return tmp_seq  # (time_len, n_dcpair, n_frame)


def read_data_gat(data_file, a_file, data_config, n_dcpair, n_his, n_pred, day_slot):
    n_train, n_val, n_test = data_config

    a_data = pd.read_csv(a_file, header=None).values
    data_seq = pd.read_csv(data_file, header=None).values
    print('V shape: ', data_seq.shape)
    print('A shape: ', a_data.shape)

    data_seq = np.log(data_seq)

    # seq_train, seq_val, seq_test: np.array, [time_len, n_dcpair, n_frame].
    seq_train = seq_gen_gat(n_train, data_seq, 0, n_his, n_pred, n_dcpair, day_slot)
    seq_val = seq_gen_gat(n_val, data_seq, n_train, n_his, n_pred, n_dcpair, day_slot)
    seq_test = seq_gen_gat(n_test, data_seq, n_train + n_val, n_his, n_pred, n_dcpair, day_slot)

    seq_stats = {'mean': np.mean(seq_train), 'std': np.std(seq_train)}

    x_train = z_score(seq_train[:, :, 0:n_his], seq_stats['mean'], seq_stats['std'])
    x_val = z_score(seq_val[:, :, 0:n_his], seq_stats['mean'], seq_stats['std'])
    x_test = z_score(seq_test[:, :, 0:n_his], seq_stats['mean'], seq_stats['std'])
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}

    y_train = z_score(seq_train[:, :, n_his:n_his+n_pred], seq_stats['mean'], seq_stats['std'])
    y_val = z_score(seq_val[:, :, n_his:n_his+n_pred], seq_stats['mean'], seq_stats['std'])
    y_test = z_score(seq_test[:, :, n_his:n_his+n_pred], seq_stats['mean'], seq_stats['std'])
    y_data = {'train': y_train, 'val': y_val, 'test': y_test}

    return [x_data, y_data, a_data, seq_stats]


def gat_model_generate(x_in, a_in, dropout, l2_reg, num_heads, channel):
    do_1 = Dropout(dropout)(x_in)
    gc_1 = GATConv(channel[0],
                   attn_heads=num_heads,
                   concat_heads=True,
                   dropout_rate=dropout,
                   activation='elu',
                   kernel_regularizer=l2(l2_reg),
                   attn_kernel_regularizer=l2(l2_reg)
                   )([do_1, a_in])
    gc_1b = GATConv(channel[0],
                   attn_heads=num_heads,
                   concat_heads=True,
                   dropout_rate=dropout,
                   activation='elu',
                   #attn_kernel_initializer='zeros',
                   return_attn_coef=True,
                   kernel_regularizer=l2(l2_reg),
                   attn_kernel_regularizer=l2(l2_reg)
                   )([do_1, a_in])
    do_2 = Dropout(dropout)(Concatenate(axis=-1)([gc_1, x_in]))
    out = Dense(1,
                   # activation='sigmoid',
                   kernel_regularizer=l2(l2_reg),
                   )(do_2)
    model = Model(inputs=[x_in, a_in], outputs=out)
    model_vis = Model(inputs=[x_in, a_in], outputs=gc_1b)
    return model, model_vis


def model_train_gat(x_data, y_data, a_data, n_dcpair, n_his, n_pred, epoch, batch_size, dropout, l2_reg, learning_rate, num_heads, channel, model_file):
    x_train = x_data['train'][:, :, 0:n_his]
    y_train = y_data['train'][:, :, 0:n_pred]
    x_val = x_data['val'][:, :, 0:n_his]
    y_val = y_data['val'][:, :, 0:n_pred]
    a_train = np.array([a_data for _ in range(y_train.shape[0])])
    a_val = np.array([a_data for _ in range(y_val.shape[0])])

    x_in = Input(shape=(n_dcpair, n_his))
    a_in = Input(shape=(n_dcpair, n_dcpair), sparse=False)

    model, model_vis = gat_model_generate(x_in, a_in, dropout, l2_reg, num_heads, channel)
    optimizer = Adam(lr=learning_rate)
    # optimizer = RMSprop(lr=learning_rate)

    model.compile(optimizer=optimizer, loss=my_cost_loss)
    # model.compile(optimizer=optimizer, loss='mse')

    model.fit([x_train, a_train],
              y_train,
              validation_data=([x_val, a_val], y_val),
              epochs=epoch,
              batch_size=batch_size,
              verbose=2,
              shuffle=True)
    model.save_weights(model_file)
    print('Training model finished!')


def model_test_gat(x_data, y_data, a_data, x_stats, n_route, n_his, n_pred, dropout, l2_reg, learning_rate, num_heads, channel, model_file, results_file):
    x_test = x_data['test'][:, :, 0:n_his]
    y_test = y_data['test'][:, :, 0:n_pred]
    a_test = np.array([a_data for _ in range(y_test.shape[0])])

    x_in = Input(shape=(n_route, n_his))
    a_in = Input(shape=(n_route, n_route), sparse=False)

    model, model_vis = gat_model_generate(x_in, a_in, dropout, l2_reg, num_heads, channel)
    optimizer = Adam(lr=learning_rate)
    #optimizer = RMSprop(lr=learning_rate)
    model.compile(optimizer=optimizer, loss='mse')
    #model.compile(optimizer=optimizer, loss=my_cost_loss)
    # model.summary()

    model.load_weights(model_file)

    data_seq = np.copy(x_test)

    len_seq, n, _ = data_seq.shape
    y_pred = np.zeros((len_seq, n, n_pred))

    pred = model.predict([data_seq[:, :, 0:n_his], a_test])
    y_pred[:, :, j] = pred[:, :, 0]

    [evl_test, results] = evaluation(y_test, y_pred, x_stats)
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

    num_heads = 8
    channel = [8]

    epoch = 100
    batch_size = 32
    dropout = 0.0          # Dropout rate for the features and adjacency matrix
    l2_reg = 5e-6          # L2 regularization rate
    learning_rate = 1e-3


    data_file = 'file_path_traffic_data'  # your file path of traffic data
    a_file = 'file_path_weight_matrix'  # your file path of attention matrix A
    model_file = 'file_path_save_model'  # save trained model
    results_file = 'file_path_results'  # save predict results

    [x_data, y_data, a_data, x_stats] = read_data_gat(data_file, a_file, data_config, n_dcpair, n_his, n_pred, day_slot)
    print('Training model:')
    start = time.time()
    model_train_gat(x_data, y_data, a_data, n_dcpair, n_his, n_pred, epoch, batch_size, dropout, l2_reg, learning_rate, num_heads, channel, model_file)
    end = time.time()
    run_time = end - start
    print('Training time: ', run_time)
    print('Inference:')
    start = time.time()
    model_test_gat(x_data, y_data, a_data, x_stats, n_dcpair, n_his, n_pred, dropout, l2_reg, learning_rate, num_heads, channel, model_file, results_file)
    end = time.time()
    run_time = end - start
    print('Inference time: ', run_time)



