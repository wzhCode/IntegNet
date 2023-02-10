# SES model: EWMA

import numpy as np
import csv
import pandas as pd
import time
from statsmodels.tsa.holtwinters import SimpleExpSmoothing

from utils.math_utils import Cost, headroom
from models.tester import save_results


def seq_gen_ses(len_seq, data_seq, offset, n_his, n_pred, n_dcpair, day_slot):
    n_frame = n_his + n_pred

    n_slot = day_slot - n_frame + 1

    tmp_seq = np.zeros((len_seq * n_slot, n_dcpair, n_frame))
    for i in range(len_seq):
        for j in range(n_slot):
            sta = (i + offset) * day_slot + j
            end = sta + n_frame
            tmp_seq[i * n_slot + j, :, :] = np.reshape(data_seq[sta:end, :].T, [n_dcpair, n_frame])

    return tmp_seq  # (time_len, n_dcpair, n_frame)

def read_data_ses(file_path, data_config, n_dcpair, n_his, n_pred, day_slot):
    n_train, n_val, n_test = data_config

    data_seq = pd.read_csv(file_path, header=None).values
    print('V shape: ', data_seq.shape)

    # seq_test: np.array, [time_len, n_dcpair, n_frame].
    seq_test = seq_gen_ses(n_test, data_seq, n_train + n_val, n_his, n_pred, n_dcpair, day_slot)

    return seq_test

def ses_predict(data, alpha):
    predict = []
    n = data.shape[0]
    for i in range(n):
        model = SimpleExpSmoothing(data[i, :]).fit(smoothing_level=alpha, optimized=False)
        #model = SimpleExpSmoothing(data[:,i]).fit(optimized = True)
        result = model.forecast(1)[0]
        predict.append(result)

    return np.array(predict)

def model_test_ses(data, n_his, n_pred, results_file):
    test_data = data

    test_X = test_data[:, :, 0:n_his]
    test_Y = test_data[:, :, n_his:n_his + n_pred]

    len_seq, n, _ = test_X.shape
    predict_Y = np.zeros((len_seq, n, n_pred))
    for index in range(len_seq):
        predict_Y[index, :, 0] = ses_predict(test_X[index, :, 0:n_his], 0.2)

    v = test_Y
    v_ = predict_Y
    head_v_ = headroom(v, v_)

    evl_test = Cost(v, head_v_)
    results = np.array([v, head_v_])

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

    data_file = 'file_path_traffic_data'  # your file path of traffic data
    results_file = 'file_path_results'  # save predict results

    data = read_data_ses(data_file, data_config, n_dcpair, n_his, n_pred, day_slot)
    print('Inference:')
    start = time.time()
    model_test_ses(data, n_his, n_pred, results_file)
    end = time.time()
    run_time = end - start
    print('Inference time: ', run_time)
