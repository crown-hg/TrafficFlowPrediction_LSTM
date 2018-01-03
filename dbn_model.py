from tfrbm import BBRBM, GBRBM


def dbn(rbm_hidden_num, rbm_visible_size, rbm_hidden_size, rbm_x, rbm_type='GBRBM'):
    weights = []
    biases = []
    for i in range(rbm_hidden_num):
        # 训练rbm
        if i == 0 and rbm_type == 'GBRBM':
            rbm = GBRBM(n_visible=rbm_visible_size, n_hidden=rbm_hidden_size, learning_rate=0.01, momentum=0.95,
                        use_tqdm=False)
        else:
            rbm = BBRBM(n_visible=rbm_visible_size, n_hidden=rbm_hidden_size, learning_rate=0.01, momentum=0.95,
                        use_tqdm=False)
        errs = rbm.fit(rbm_x, n_epoches=10, batch_size=100, verbose=True)
        rbm_x = rbm.transform(rbm_x)
        rbm_w, vb, rbm_b = rbm.get_weights()
        rbm_visible_size = rbm_hidden_size
        weights.append(rbm_w)
        biases.append(rbm_b)
    return weights, biases
