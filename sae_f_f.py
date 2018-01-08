import tensorflow as tf
import datetime
from get_data import *
from nn_model import nn
from train_log import print_to_console
from sae_model import sae

gpu_device = 0

# 超参数
time_step = 3
train_num = 71 * 96
test_num = 18 * 96
hidden_num = 2
hidden_size = 300
act_function = None
LR = 0.0001
hide_function = tf.nn.relu
sae_epoches = 300
sae_lr = 0.01
max_epoch = 50000
rbm_type = 'GBRBM'
data_type = 'normal'
filename = 'result_log/2018-1-8_SAE.txt'

# 取数据
data_x = flow_normalized
data_y = flow_normalized

train_x, train_y, test_x, test_y = create_train_test(data_x, data_y, time_step, train_num, test_num)
batch_size, _, input_size = train_x.shape
_, output_size = train_y.shape
train_x = np.reshape(train_x, (-1, time_step * input_size))
test_x = np.reshape(test_x, (-1, time_step * input_size))

with tf.device('/gpu:%d' % gpu_device):
    # sae初始化权值
    weights, biases = sae(train_x, hidden_num, hidden_size, sae_epoches, lr=sae_lr, hide_function=hide_function)

    # 根据初始化的权值来构建神经网络
    x_input = tf.placeholder(tf.float32, [None, time_step * input_size], name='dbn_x')
    y_real = tf.placeholder(tf.float32, [None, output_size], name='dbn_y')
    hide_output = nn(x_input, hidden_num, hidden_size, hide_act_function=hide_function, weights=weights, biases=biases)
    y_pred = tf.layers.dense(hide_output, output_size)
    mse = tf.losses.mean_squared_error(y_real, y_pred)
    train_op = tf.train.AdamOptimizer(LR).minimize(mse)

config = tf.ConfigProto(allow_soft_placement=True, log_device_placement=True)
# 设置 GPU 按需增长
config.gpu_options.allow_growth = True
sess = tf.Session(config=config)
# 初始化变量
sess.run(tf.global_variables_initializer())

mre_result = []
mae_result = []
rmse_result = []

with open(filename, 'a') as fp:
    fp.write('\n%s\t%s' % (rbm_type, data_type))
# 训练和测试
start_time = datetime.datetime.now()
for i in range(1, max_epoch + 1):
    sess.run(train_op, feed_dict={x_input: train_x, y_real: train_y})
    if i % 50 == 0:
        feed_dict = {x_input: train_x, y_real: train_y}
        train_y_pred = sess.run(y_pred, feed_dict=feed_dict)
        feed_dict = {x_input: test_x, y_real: test_y}
        test_y_pred = sess.run(y_pred, feed_dict=feed_dict)
        mre, mae, rmse = print_to_console(i, train_y, train_y_pred, test_y, test_y_pred, flow_min, flow_max)
        mre_result.append(mre)
        mae_result.append(mae)
        rmse_result.append(rmse)
        # if i % 2000 == 0:
        #     current_time = time.strftime('%Y-%m-%d %H:%M', time.localtime(time.time()))
        #     end_time = datetime.datetime.now()
        #     time_spend = (end_time - start_time).seconds
        #     start_time = datetime.datetime.now()
        #     test_result = '\n%s\t%d\t%d\t%.4f\t%d\t%d\t%.4f\t%.2f\t%.2f\t%s' % (
        #         current_time, rbm_hidden_num, rbm_hidden_size, LR, time_step, i,
        #         mre, mae, rmse, time_spend)
        #     with open(filename, 'a') as fp:
        #         fp.write(test_result)
min_index = mre_result.index(np.min(mre_result))
line = '\nepoch=%d min_mre %.4f %.2f %.2f' % (
    (min_index + 1) * 50, mre_result[min_index], mae_result[min_index], rmse_result[min_index])
# with open(filename, 'a') as fp:
#     fp.write(line)
print(line)
sess.close()
