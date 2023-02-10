
import time
import numpy as np
import tensorflow as tf

from data_loader.data_utils import gen_batch
from models.tester import model_inference
from models.integnet_model import *


def model_train(inputs, blocks, args, save_model_name):
    '''
    Train the model.
    '''
    n, n_his, n_pred = args.n_dcpair, args.n_his, args.n_pred
    Ks, Kt = args.ks, args.kt
    batch_size, epoch, opt = args.batch_size, args.epoch, args.opt

    x = tf.placeholder(tf.float32, [None, n_his + n_pred, n, 1], name='data_input')
    keep_prob = tf.placeholder(tf.float32, name='keep_prob')
    is_train = tf.placeholder_with_default(False, (), 'is_train')  # opt: batch normalization

    # build model
    train_loss, pred = build_model_tcn_gcn(x, n_his, Ks, Kt, blocks, keep_prob, is_train)

    # Learning rate settings
    global_steps = tf.Variable(0, trainable=False)
    len_train = inputs.get_len('train')
    if len_train % batch_size == 0:
        epoch_step = len_train / batch_size
    else:
        epoch_step = int(len_train / batch_size) + 1
    lr = tf.train.exponential_decay(args.lr, global_steps, decay_steps=10 * epoch_step, decay_rate=0.7, staircase=True)
    step_op = tf.assign_add(global_steps, 1)
    with tf.control_dependencies([step_op]):
        train_op = tf.train.RMSPropOptimizer(lr).minimize(train_loss)


    update_op = tf.get_collection(tf.GraphKeys.UPDATE_OPS)  # opt: BN
    train_op = tf.group([train_op, update_op])  # opt: BN

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        min_te_val = min_va_val = np.array([1e5, 1e5, 1e5])

        for i in range(epoch):
            start_time = time.time()

            for j, x_batch in enumerate(gen_batch(inputs.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
                sess.run(train_op, feed_dict={x: x_batch[:, 0:n_his + n_pred, :, :], keep_prob: 1.0, is_train: False})  # opt: BN
            print(f'Epoch {i:2d} Training Time {time.time() - start_time:.3f}s')

            start_time = time.time()
            min_va_val, min_te_val = model_inference(sess, pred, inputs, batch_size, n_his, n_pred, min_va_val, min_te_val)

            va, te = min_va_val, min_te_val
            print(f'Evaluation: [val, test]'
                  f'Overcost {va[0]:6.3f}, {te[0]:6.3f}; '
                  f'Undercost  {va[1]:6.3f}, {te[1]:6.3f}; '
                  f'Cost  {va[2]:6.3f}, {te[2]:6.3f}. ')
            print(f'Epoch {i:2d} Inference Time {time.time() - start_time:.3f}s')

            if (i + 1) % args.save == 0:
                model_save(sess, global_steps, save_model_name)
    print('Training model finished!')
