import numpy as np
from get_data import inverse_normalize, inverse_standardize


def get_metrics(real, pred):
    mre = np.mean(np.abs(real - pred) / real)
    mae = np.mean(np.abs(real - pred))
    rmse = np.sqrt(np.mean(np.square(real - pred)))
    return mre, mae, rmse


def get_metrics_normal(y, pred_y, y_min, y_max):
    real = inverse_normalize(y, y_min, y_max)
    pred = inverse_normalize(pred_y, y_min, y_max)
    return get_metrics(real, pred)


def get_metrics_standard(y, pred_y, y_min, y_max):
    real = inverse_standardize(y, y_min, y_max)
    pred = inverse_standardize(pred_y, y_min, y_max)
    return get_metrics(real, pred)


def print_to_console(i, train_y, train_y_pred, test_y, test_y_pred, y_min, y_max, data_tye='normal'):
    if data_tye == 'normal':
        train_mre, train_mae, train_rmse = get_metrics_normal(train_y, train_y_pred, y_min, y_max)
        test_mre, test_mae, test_rmse = get_metrics_normal(test_y, test_y_pred, y_min, y_max)
    else:
        train_mre, train_mae, train_rmse = get_metrics_standard(train_y, train_y_pred, y_min, y_max)
        test_mre, test_mae, test_rmse = get_metrics_standard(test_y, test_y_pred, y_min, y_max)
    print('epoch %d  train %.4f %.2f %.2f test %.4f %.2f %.2f' %
          (i, train_mre, train_mae, train_rmse, test_mre, test_mae, test_rmse))
    return test_mre, test_mae, test_rmse
