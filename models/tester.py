
import tensorflow as tf
import numpy as np
import time

from data_loader.data_utils import gen_batch
from utils.math_utils import evaluation
from os.path import join as pjoin


def multi_pred(sess, y_pred, seq, batch_size, n_his, n_pred, dynamic_batch=True):
    '''
    Multi_prediction function.
    '''
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        test_seq = np.copy(i[:, 0:n_his + n_pred, :, :])
        pred = sess.run(y_pred, feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0})
        #pred = sess.run(y_pred, feed_dict={'data_input:0': test_seq, 'keep_prob:0': 1.0, 'is_train:0': False})  # opt: BN
        if isinstance(pred, list):
            pred = np.array(pred[0])
        pred_list.append(pred)
    pred_array = np.concatenate(pred_list, axis=0)
    # pred_array: [len_seq, n_dcpair, channel_size]

    return pred_array, pred_array.shape[0]


def model_inference(sess, pred, inputs, batch_size, n_his, n_pred, min_va_val, min_te_val):
    '''
    Model inference function.
    '''
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()

    y_val, len_val = multi_pred(sess, pred, x_val, batch_size, n_his, n_pred)
    [evl_val, results] = evaluation(x_val[0:len_val, n_his, :, :], y_val, x_stats)

    if evl_val[2] < min_va_val[2]:
        min_va_val = evl_val
        y_pred, len_pred = multi_pred(sess, pred, x_test, batch_size, n_his, n_pred)
        [evl_pred, results] = evaluation(x_test[0:len_pred, n_his, :, :], y_pred, x_stats)
        min_te_val = evl_pred
    return min_va_val, min_te_val


def save_results(results, file_path):
    '''
    Save predict results.
    '''
    print(f'shape of predict results: {results.shape}')
    with open(file_path, 'w+') as f:
        np.save(f, results)
    f.close()


def model_test(inputs, batch_size, n_his, n_pred, file_path, load_path):
    '''
    Load and test saved model from the checkpoint.
    '''
    model_path = tf.train.get_checkpoint_state(load_path).model_checkpoint_path

    test_graph = tf.Graph()

    with test_graph.as_default():
        saver = tf.train.import_meta_graph(pjoin(f'{model_path}.meta'))

    with tf.Session(graph=test_graph) as test_sess:
        saver.restore(test_sess, tf.train.latest_checkpoint(load_path))
        print(f'>> Loading saved model from {model_path} ...')

        pred = test_graph.get_collection('y_pred')

        x_test, x_stats = inputs.get_data('test'), inputs.get_stats()

        y_test, len_test = multi_pred(test_sess, pred, x_test, batch_size, n_his, n_pred)
        [evl_test, results] = evaluation(x_test[0:len_test, n_his, :, :], y_test, x_stats)

        # save predict results
        save_results(results, file_path)

        te = evl_test
        print(f'Evaluation: Overcost {te[0]:6.3f}; Undercost {te[1]:6.3f}; Cost {te[2]:6.3f}. ')

    print('Testing model finished!')
