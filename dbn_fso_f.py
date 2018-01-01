import numpy as np
import tensorflow as tf
from tfrbm import BBRBM, GBRBM
import datetime
import time
from get_data import *

gpu_device = 0

# 超参数
time_step = 4
train_num = 71 * 96
test_num = 18 * 96
rbm_hidden_num = 1
rbm_hidden_size = 200
act_function = None
LR = 0.001
max_epoch = 1000
rbm_type = 'GBRBM'
filename = 'result_log/GBRBM_DBN.txt'

# 取数据
if rbm_type == 'GBRBM':
    x = np.hstack((flow_standardized, speed_standardized, occupancy_standardized))
    y = flow_standardized
else:
    x = np.hstack((flow_normalized, speed_normalized, occupancy_normalized))
    y = flow_normalized

train_x, train_y, test_x, test_y = create_train_test(x, y, time_step, train_num, test_num)

for rbm_hidden_num in [1, 2, 3]:
    for rbm_hidden_size in [100, 200, 300, 400, 500]:
        batch_size, time_step, input_size = train_x.shape
        _, output_size = train_y.shape
        train_X = np.reshape(train_x, (-1, time_step * input_size))
        train_Y = train_y
        test_X = np.reshape(test_x, (-1, time_step * input_size))
        test_Y = test_y
        rbm_x = train_X
        rbm_visible_size = time_step * input_size
        weights = []
        biases = []

        with tf.device('/gpu:%d' % gpu_device):
            for i in range(rbm_hidden_num):
                # 训练rbm
                if rbm_type == 'GBRBM':
                    rbm = GBRBM(n_visible=rbm_visible_size, n_hidden=rbm_hidden_size, learning_rate=0.01, momentum=0.95,
                                use_tqdm=False)
                else:
                    rbm = BBRBM(n_visible=rbm_visible_size, n_hidden=rbm_hidden_size, learning_rate=0.01, momentum=0.95,
                                use_tqdm=False)
                errs = rbm.fit(rbm_x, n_epoches=10, batch_size=100, verbose=True)
                rbm_x = rbm.transform(rbm_x)
                rbm_W, vb, rbm_b = rbm.get_weights()
                rbm_visible_size = rbm_hidden_size
                weights.append(rbm_W)
                biases.append(rbm_b)

            with tf.name_scope('dbn'):
                dbn_x = tf.placeholder(tf.float32, [None, time_step * input_size], name='dbn_x')
                dbn_y = tf.placeholder(tf.float32, [None, output_size], name='dbn_y')
                dbn_batch_size = tf.placeholder(tf.float32, [], name='dbn_batch_size')

            output = list()

            with tf.name_scope('input_layer'):
                input_layer_output = tf.layers.dense(dbn_x, rbm_hidden_size,
                                                     kernel_initializer=tf.constant_initializer(weights[0]),
                                                     bias_initializer=tf.constant_initializer(biases[0]),
                                                     activation=tf.nn.sigmoid)
                output.append(input_layer_output)
            with tf.name_scope('hide_layer'):
                for index in range(1, rbm_hidden_num, 1):
                    hide_layer_output = tf.layers.dense(output[index - 1], rbm_hidden_size,
                                                        kernel_initializer=tf.constant_initializer(weights[index]),
                                                        bias_initializer=tf.constant_initializer(biases[index]),
                                                        activation=tf.nn.sigmoid)
                    output.append(hide_layer_output)
            with tf.name_scope('output_layer'):
                output_layer_output = tf.layers.dense(output[rbm_hidden_num - 1], output_size, activation=act_function)
                output.append(output_layer_output)
            dbn_y_pre = output[-1]
            mse = tf.losses.mean_squared_error(dbn_y, dbn_y_pre)
            train_op = tf.train.AdamOptimizer(LR).minimize(mse)

            # mre = tf.reduce_mean(tf.div(tf.abs(tf.subtract(dbn_y, real_pre)), dbn_y))
            # mae = tf.reduce_mean(tf.abs(tf.subtract(dbn_y, real_pre)))
            # rmse = tf.sqrt(mse)

        config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
        # 设置 GPU 按需增长
        config.gpu_options.allow_growth = True
        sess = tf.Session(config=config)
        # 初始化变量
        sess.run(tf.global_variables_initializer())


        def get_metrics(real, pred):
            miss_data_position = real <= 0
            real[miss_data_position] = 1
            pred[miss_data_position] = 1
            mre = np.mean(np.abs(real - pred) / real)
            mae = np.mean(np.abs(real - pred))
            rmse = np.sqrt(np.mean(np.square(real - pred)))
            return mre, mae, rmse


        def get_metrics_normal(x, y, bs, y_min, y_max):
            feed_dict = {dbn_x: x, dbn_y: y, dbn_batch_size: bs}
            pred_y = sess.run(dbn_y_pre, feed_dict=feed_dict)
            real = inverse_normalize(y, y_min, y_max)
            pred = inverse_normalize(pred_y, y_min, y_max)
            return get_metrics(real, pred)


        def get_metrics_standard(x, y, bs, y_mean, y_std):
            feed_dict = {dbn_x: x, dbn_y: y, dbn_batch_size: bs}
            pred_y = sess.run(dbn_y_pre, feed_dict=feed_dict)
            real = inverse_standardize(y, y_mean, y_std)
            pred = inverse_standardize(pred_y, y_mean, y_std)
            return get_metrics(real, pred)


        train_mre_result = []
        test_mre_result = []
        test_mae_result = []
        test_rmse_result = []

        with open(filename, 'a') as fp:
            fp.write('\n%s' % rbm_type)
        # 训练和测试
        start_time = datetime.datetime.now()
        for i in range(1, max_epoch + 1):
            sess.run(train_op, feed_dict={dbn_x: train_X, dbn_y: train_Y, dbn_batch_size: train_num})
            if i % 50 == 0:
                if rbm_type == 'GBRBM':
                    train_mre, train_mae, train_rmse = get_metrics_standard(train_X, train_Y, train_num, flow_mean,
                                                                            flow_std)
                    test_mre, test_mae, test_rmse = get_metrics_standard(test_X, test_Y, test_num, flow_mean, flow_std)
                else:
                    train_mre, train_mae, train_rmse = get_metrics_normal(train_X, train_Y, train_num, flow_min,
                                                                          flow_max)
                    test_mre, test_mae, test_rmse = get_metrics_normal(test_X, test_Y, test_num, flow_min, flow_max)
                train_mre_result.append(train_mre)
                test_mre_result.append(test_mre)
                test_mae_result.append(test_mae)
                test_rmse_result.append(test_rmse)
                print('epoch %d train %.4f %.2f %.2f test %.4f %.2f %.2f test' % (
                    i, train_mre, train_mae, train_rmse, test_mre, test_mae, test_rmse))
                if i % 2000 == 0:
                    current_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
                    end_time = datetime.datetime.now()
                    time_spend = (end_time - start_time).seconds
                    start_time = datetime.datetime.now()
                    test_result = '\n%s\t%d\t%d\t\t%d\t%d\t%.4f\t%.2f\t%.2f\t%s' % (
                        current_time, rbm_hidden_num, rbm_hidden_size, time_step, i,
                        test_mre, test_mae, test_rmse, time_spend)
                    with open(filename, 'a') as fp:
                        fp.write(test_result)
        with open(filename, 'a') as fp:
            min_index = test_mre_result.index(np.min(test_mre_result))
            line = '\nepoch=%d min_mre %.4f %.2f %.2f' % (
                (min_index + 1) * 50, test_mre_result[min_index], test_mae_result[min_index],
                test_rmse_result[min_index])
            fp.write(line)
        print(line)
        sess.close()
