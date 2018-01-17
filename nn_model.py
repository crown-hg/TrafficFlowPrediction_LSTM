import tensorflow as tf


def nn(x_input, hidden_num, hidden_size, hide_act_function=None, weights=None, biases=None):
    # 产生一个只有 输入层 和 隐藏层 的神经网络
    # x_input必须是tensor
    output = list()
    with tf.name_scope('input_layer'):
        if weights is None:
            input_layer_output = tf.layers.dense(x_input, hidden_size, activation=hide_act_function)
        else:
            input_layer_output = tf.layers.dense(x_input, hidden_size,
                                                 kernel_initializer=tf.constant_initializer(weights[0]),
                                                 bias_initializer=tf.constant_initializer(biases[0]),
                                                 activation=hide_act_function)
        output.append(input_layer_output)

    with tf.name_scope('hide_layer'):
        for index in range(1, hidden_num, 1):
            if weights is None:
                hide_layer_output = tf.layers.dense(x_input, hidden_size, activation=hide_act_function)
            else:
                hide_layer_output = tf.layers.dense(output[index - 1], hidden_size,
                                                    kernel_initializer=tf.constant_initializer(weights[index]),
                                                    bias_initializer=tf.constant_initializer(biases[index]),
                                                    activation=hide_act_function)
            output.append(hide_layer_output)

    y_pre = output[-1]
    return y_pre
