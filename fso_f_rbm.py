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
gpu_device = 0

for rbm_hidden in [100, 200, 300, 400]:
    with tf.device('/gpu:%d' % gpu_device):
        # 训练rbm
        _, _, input_size = train_x.shape
        rbm = GBRBM(n_visible=input_size, n_hidden=rbm_hidden, learning_rate=0.01, momentum=0.95, use_tqdm=True)
        errs = rbm.fit(train_x[:, 0, :], n_epoches=10, batch_size=100, verbose=True)
        rbm_W, vb, rbm_b = rbm.get_weights()

    # lstm的hyper-parameters
    # hidden_size = 400
    layer_num = 1
    max_epoch = 20000
    dropout_keep_rate = 0.9

    # 存储运行结果的文件
    # date_time = time.strftime('%Y-%m-%d', time.localtime(time.time()))
    file_name = 'fso_f_lstm_gbrbm'
    train_mre_result = []
    test_mre_result = []
    test_mae_result = []
    test_rmse_result = []
    for hidden_size in [100, 200, 300, 400, 500, 600]:
        train_mre, test_mre, test_mae, test_rmse = lstm_test(hidden_size, layer_num, max_epoch, dropout_keep_rate,
                                                             train_x, train_y, test_x, test_y, file_name,
                                                             use_rbm=True, rbm_w=rbm_W, rbm_b=rbm_b,
                                                             gpu_device=gpu_device)
        train_mre_result.append(train_mre)
        test_mre_result.append(test_mre)
        test_mae_result.append(test_mae)
        test_rmse_result.append(test_rmse)
