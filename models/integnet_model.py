# IntegNet model

import tensorflow as tf


def gconv(x, theta, Ks, c_in, c_out):
    # Refer to https://github.com/VeritasYin/STGCN_IJCAI-18.git
    '''
    Spectral-based graph convolution function.
    return: tensor, [batch_size, n_dcpair, c_out].
    '''
    # graph kernel: tensor, [n_route, Ks*n_route]
    kernel = tf.get_collection('graph_kernel')[0]
    n = tf.shape(kernel)[0]
    # x -> [batch_size, c_in, n_route] -> [batch_size*c_in, n_route]
    x_tmp = tf.reshape(tf.transpose(x, [0, 2, 1]), [-1, n])
    # x_mul = x_tmp * ker -> [batch_size*c_in, Ks*n_route] -> [batch_size, c_in, Ks, n_route]
    x_mul = tf.reshape(tf.matmul(x_tmp, kernel), [-1, c_in, Ks, n])
    # x_ker -> [batch_size, n_route, c_in, K_s] -> [batch_size*n_route, c_in*Ks]
    x_ker = tf.reshape(tf.transpose(x_mul, [0, 3, 1, 2]), [-1, c_in * Ks])
    # x_gconv -> [batch_size*n_route, c_out] -> [batch_size, n_route, c_out]
    x_gconv = tf.reshape(tf.matmul(x_ker, theta), [-1, n, c_out])
    return x_gconv


def my_cost_loss(y, y_):
    diff = y_ - y
    return tf.nn.l2_loss(diff)


def temporal_conv_layer(x, Kt, c_in, c_out, act_func):
    '''
    Temporal convolution layer.
    return: tensor, [batch_size, time_step-Kt+1, n_dcpair, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()

    if c_in < c_out:
        x_input = tf.concat([x, tf.zeros([tf.shape(x)[0], T, n, c_out - c_in])], axis=3)
    elif c_in == c_out:
        x_input = x
    else:
        raise ValueError(f'ERROR: c_in must be not greater than c_out here.')

    x_input = x_input[:, Kt - 1:T, :, :]

    if act_func == 'GLU':
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, 2 * c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([2 * c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return (x_conv[:, :, :, 0:c_out] + x_input) * tf.nn.sigmoid(x_conv[:, :, :, -c_out:])
    elif act_func == 'sigmoid':
        wt = tf.get_variable(name='wt', shape=[Kt, 1, c_in, c_out], dtype=tf.float32)
        tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(wt))
        bt = tf.get_variable(name='bt', initializer=tf.zeros([c_out]), dtype=tf.float32)
        x_conv = tf.nn.conv2d(x, wt, strides=[1, 1, 1, 1], padding='VALID') + bt
        return tf.nn.sigmoid(x_conv)
    else:
        raise ValueError(f'ERROR: activation function "{act_func}" is not defined.')


def graph_conv_layer(x, Ks, c_in, c_out, is_train):
    '''
    Graph convolution layer.
    return: tensor, [batch_size, time_step, n_dcpair, c_out].
    '''
    _, T, n, _ = x.get_shape().as_list()
    x_input = x

    ws = tf.get_variable(name='wg', shape=[Ks * c_in, c_out], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(ws))
    bs = tf.get_variable(name='bg', initializer=tf.zeros([c_out]), dtype=tf.float32)

    x_gconv = gconv(tf.reshape(x, [-1, n, c_in]), ws, Ks, c_in, c_out) + bs

    # x_gconv = tf.layers.batch_normalization(x_gconv, momentum=0.99, epsilon=1e-3, training=is_train) # opt: BN

    x_gcn = tf.reshape(x_gconv, [-1, T, n, c_out])

    return tf.nn.relu(x_gcn[:, :, :, 0:c_out] + x_input)


def fully_con_layer(x, n, channel, scope):
    '''
    Fully connected layer.
    return: [batch_size, 1, n_dcpair, 1].
    '''
    w = tf.get_variable(name=f'w_{scope}', shape=[1, 1, channel, 1], dtype=tf.float32)
    tf.add_to_collection(name='weight_decay', value=tf.nn.l2_loss(w))
    b = tf.get_variable(name=f'b_{scope}', initializer=tf.zeros([n, 1]), dtype=tf.float32)
    return tf.nn.conv2d(x, w, strides=[1, 1, 1, 1], padding='SAME') + b


def output_layer(x, T, scope):
    '''
    Output layer.
    return: tensor, [batch_size, 1, n_dcpair, 1].
    '''
    _, _, n, channel = x.get_shape().as_list()

    # maps multi-steps to one.
    with tf.variable_scope(f'{scope}_step'):
        x_o = temporal_conv_layer(x, T, channel, channel, act_func='sigmoid')

    # maps multi-channels to one.
    x_fc = fully_con_layer(x_o, n, channel, scope)
    return x_fc


def tcn_gcn_block(x, Ks, Kt, channels, scope, keep_prob, is_train):
    '''
    TCN + GCN block.
    return: tensor, [batch_size, time_step, n_dcpair, c_out].
    '''
    c_in, c_t, c_out = channels

    with tf.variable_scope(f'tcn_gcn_block_{scope}'):
        # x: [batch_size, time_step, n_dcpair, c_in]
        x_t = temporal_conv_layer(x, Kt, c_in, c_t, act_func='GLU')
        # x_t: [batch_size, time_step - Kt + 1, n_dcpair, c_t]
        x_g = graph_conv_layer(x_t, Ks, c_t, c_out, is_train)  # opt: BN
        # x_g: [batch_size, time_step - Kt + 1, n_dcpair, c_out]

    return tf.nn.dropout(x_g, keep_prob)


def build_model_tcn_gcn(inputs, n_his, Ks, Kt, blocks, keep_prob, is_train):
    '''
    Build the tcn + gcn model.
    '''
    x = inputs[:, 0:n_his, :, :]

    for i, channels in enumerate(blocks):
        x = tcn_gcn_block(x, Ks, Kt, channels, i, keep_prob, is_train)

    # Output Layer
    Ko = x.get_shape()[1]
    y = output_layer(x, Ko, 'output_layer')

    train_loss = my_cost_loss(inputs[:, n_his:n_his + 1, :, :], y)

    single_pred = y[:, 0, :, :]
    tf.add_to_collection(name='y_pred', value=single_pred)
    return train_loss, single_pred


def model_save(sess, global_steps, model_name):
    '''
    Save the checkpoint of trained model.
    '''
    saver = tf.train.Saver(max_to_keep=3)
    prefix_path = saver.save(sess, model_name, global_step=global_steps)
    print(f'<< Saving model to {prefix_path} ...')

