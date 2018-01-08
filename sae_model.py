import tensorflow as tf
import numpy as np


def auto_encoder(ae_x, hidden_size, n_epoches, lr=0.01, act_function=None):
    input_size = ae_x.shape[1]
    in_x = tf.placeholder(dtype=tf.float32, shape=ae_x.shape)
    w_in = tf.Variable(tf.truncated_normal([input_size, hidden_size], stddev=0.1), dtype=tf.float32)
    b_in = tf.Variable(tf.constant(0.1, shape=[hidden_size]), dtype=tf.float32)
    xw_b_in = tf.add(tf.matmul(in_x, w_in), b_in)
    if act_function is None:
        hide_x = xw_b_in
    else:
        hide_x = act_function(xw_b_in)
    w_hide = tf.Variable(tf.truncated_normal([hidden_size, input_size], stddev=0.1), dtype=tf.float32)
    b_hide = tf.Variable(tf.constant(0.1, shape=[input_size]), dtype=tf.float32)
    xw_b_hide = tf.add(tf.matmul(hide_x, w_hide), b_hide)
    if act_function is None:
        x_pre = xw_b_hide
    else:
        x_pre = act_function(xw_b_hide)
    mse = tf.losses.mean_squared_error(in_x, x_pre)
    train_op = tf.train.AdamOptimizer(lr).minimize(mse)

    config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    sess.run(tf.global_variables_initializer())

    for i in range(n_epoches):
        sess.run(train_op, feed_dict={in_x: ae_x})
        err = sess.run(mse, feed_dict={in_x: ae_x})
        print('epoche %d mse %.8f' % (i, err))

    return sess.run([w_in, b_in, hide_x], feed_dict={in_x: ae_x})


def sae(x_input, hidden_num, hidden_size, n_epoches, lr=0.01, hide_function=None):
    weights = []
    biases = []
    for i in range(hidden_num):
        # 训练rbm
        w, b, x_input = auto_encoder(x_input, hidden_size, n_epoches, lr=lr, act_function=hide_function)
        weights.append(w)
        biases.append(b)
    return weights, biases
