import numpy as np


def get_metrics(y, pred_y):
    mre = np.mean(np.abs(y - pred_y) / y)
    mae = np.mean(np.abs(y - pred_y))
    rmse = np.sqrt(np.mean(np.square(y - pred_y)))
    return mre, mae, rmse


def print_to_console(i, train_y_pred, test_y_pred, data):
    train_y_pred_real = data['data_process'].reconstruct(train_y_pred)
    train_y_real = data['data_process'].reconstruct(data['train_y'])
    test_y_pred_real = data['data_process'].reconstruct(test_y_pred)
    test_y_real = data['data_process'].reconstruct(data['test_y'])

    train_mre, train_mae, train_rmse = get_metrics(train_y_real, train_y_pred_real)
    test_mre, test_mae, test_rmse = get_metrics(test_y_real, test_y_pred_real)
    print('epoch %d  train %.4f %.2f %.2f test %.4f %.2f %.2f' %
          (i, train_mre, train_mae, train_rmse, test_mre, test_mae, test_rmse))
    return test_mre, test_mae, test_rmse

# def print_to_console(i, train_y_pred, test_y_pred, data):
#     if data_tye == 'normal':
#         train_mre, train_mae, train_rmse = get_metrics_normal(train_y, train_y_pred, y_min, y_max, train_y_aver)
#         test_mre, test_mae, test_rmse = get_metrics_normal(test_y, test_y_pred, y_min, y_max, test_y_aver)
#     else:
#         train_mre, train_mae, train_rmse = get_metrics_standard(train_y, train_y_pred, y_min, y_max)
#         test_mre, test_mae, test_rmse = get_metrics_standard(test_y, test_y_pred, y_min, y_max)
#     print('epoch %d  train %.4f %.2f %.2f test %.4f %.2f %.2f' %
#           (i, train_mre, train_mae, train_rmse, test_mre, test_mae, test_rmse))
#     return test_mre, test_mae, test_rmse
