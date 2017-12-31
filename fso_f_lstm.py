import numpy as np
import tensorflow as tf
import time
from tfrbm import BBRBM, GBRBM
from lstm_multicell import lstm_test
from get_data import train_fso_test_f

# 取数据
time_step = 4
train_num = 71 * 96
test_num = 18 * 96
train_x, train_y, test_x, test_y = train_fso_test_f(time_step, train_num, test_num)
with tf.device('/gpu:1'):
    # lstm的hyper-parameters
    # hidden_size = 300
    layer_num = 1
    max_epoch = 20000
    # dropout_keep_rate = 0.8

    # 存储运行结果的文件
    date_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    file_name = '%sfso_f_norbm' % date_time

    train_mre, test_mre, test_mae, test_rmse = lstm_test(450, layer_num, max_epoch, 1.0, train_x, train_y,
                                    test_x, test_y, file_name)
    # trm = []
    # tem = []
    # for hidden_size in range(100, 601, 50):
    #     for dropout_keep_rate in np.arange(0.8, 1.01, 0.1):
    #         train_mre, test_mre = lstm_test(hidden_size, layer_num, max_epoch, dropout_keep_rate, train_x, train_y,
    #                                         test_x, test_y, file_name)
    #         trm.append(train_mre)
    #         tem.append(tem)
