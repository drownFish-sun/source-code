import math

import numpy as np
import pandas as pd
from numpy.linalg import inv
from scipy import sparse

from hyperG import HyperG

def model_eval(actual, pred):
    confusion = pd.crosstab(actual, pred, rownames=['Actual'], colnames=['Predicted'])
    try:
        TP = confusion.loc[1, 1]
    except:
        TP = 0
    try:
        TN = confusion.loc[0, 0]
    except:
        TN = 0
    try:
        FP = confusion.loc[0, 1]
    except:
        FP = 0
    try:
        FN = confusion.loc[1, 0]
    except:
        FN = 0

    print("TP={}, TN={}, FP={}, FN={}".format(TP, TN, FP, FN))

    out = {}
    out['ACR'] = round((TP + TN) / (TP + TN + FP + FN) * 100, 4)
    try:
        out['PRE'] = round(TP / (TP + FP) * 100, 4)
    except ZeroDivisionError:
        out['PRE'] = 0
    try:
        out['PF'] = round(FP / (FP + TN) * 100, 4)
    except ZeroDivisionError:
        out['PF'] = 0
    try:
        out['PD'] = round(TP / (TP + FN) * 100, 4)
    except ZeroDivisionError:
        out['PD'] = 0
    try:
        out['F1'] = round((2 * out['PRE'] * out['PD']) / (out['PRE'] + out['PD']), 4)
    except ZeroDivisionError:
        out['F1'] = 0
    # out['DI'] = round((TP / (TP + FN)) / (TN / (TN + FP)) * 100, 4)
    # out['TPR'] = round(TP / (TP + FN) * 100, 4)
    # out['TNR'] = round(TN / (FP + TN) * 100, 4)
    return out
def cal_DF(i, F, Fi):
    ret = 0
    pi = np.argmax(Fi, axis=1)
    for j in range(len(F)):
        ret += (1 if j != i else 0) * np.sum(np.array(pi - np.argmax(F[j], axis=1)) ** 2)
    return ret
def cal_F_V(i, F, alpha):
    F_ret = np.zeros((F[i].shape[0], F[i].shape[1]))
    for j in range(len(F)):
        # F_ret += (alpha[j] if j != i else 0) * F[j]
        F_ret += (1 if j != i else 0) * F[j]
    return F_ret

def joint_dev(embedding, y_, test_y, dev_y, dev_label, id_dev, u, len_dev, lamda=1, mu=1, gama=1):
    _y = np.ones(dev_y.shape[0] + test_y.shape[0]) * -1
    y = np.concatenate([y_, _y])
    hg = [HyperG(emb, y_, u=u, K=4) for emb in embedding]
    # F = [minmax_scale(hg_.predict(), (0.0, 1.0)) for hg_ in hg]
    F = [hg_.predict() for hg_ in hg]
    # F = [hg_.predict_() for hg_ in hg]
    Y_dev = np.zeros((dev_y.shape[0], 2))
    for i in range(len(dev_y)):
        Y_dev[i, :] = 0
        Y_dev[i, dev_y[i]] = 1
    res = []
    for i in range(len(hg)):
        pre = np.argmax(F[i], axis=1)[len_dev:]
        res.append(pre)
        model_eval(test_y, pre)
        # pre = np.argmax(F[i], axis=1)[len(y_) : len(y_) + len(dev_y)]
        # model_eval(dev_y, pre)
        # VHL_ACC[i].append(reg['ACR'])
        # VHL_PRE[i].append(reg['PRE'])
        # VHL_PF[i].append(reg['PF'])
        # VHL_PD[i].append(reg['PD'])
        # VHL_F1[i].append(reg['F1'])
        # row_sums = F[i].sum(axis=1)
        # row_sums = row_sums[:, np.newaxis]
        # F[i] /= row_sums
    nt = F[0].shape[0]
    ne = 2 * nt
    I = sparse.eye(nt)
    U = [hg_.U for hg_ in hg]
    Y = hg[0].Y
    num_hg = len(hg)
    alpha = np.ones(num_hg) * (1. / num_hg)
    L2 = U
    for i in range(num_hg):
        L2[i] = hg[i].L2
        # print(np.min(L2[i]), np.max(L2[i]))
    loss = -1
    iters = 10
    # for i in range(num_hg):
    #     F[i] = Y

    pred_to_logs = []
    _pred = []
    for i in range(num_hg):
        pred = np.argmax(np.array(F[i]), axis=1)
        pred = pred[len(y_):len_dev]
        _pred.append(pred)
        p_t_l = np.zeros(dev_label.shape[0])
        for j in range(len(pred)):
            p_t_l[id_dev[j]] = pred[j]
        model_eval(p_t_l, dev_label)
        pred_to_logs.append(p_t_l)
    phi = np.zeros(num_hg)
    for i in range(num_hg):
        # phi[i] = lamda * np.sum(np.array(np.argmax(F[i], axis=1) - np.argmax(Y, axis=1))[y != -1] ** 2)
        # phi[i] = lamda * np.sum(np.array(np.argmax(F[i], axis=1)[len(y_) : len_dev] - dev_y) ** 2) \
        #          + gama * np.sum(np.array(pred_to_logs[i] - dev_label) ** 2)
        # phi[i] = 1 / np.trace(F[i][len(y_) : len(y_) + dev_y.shape[0], :].T.dot(L2[i].toarray()[len(y_) : len(y_) + dev_y.shape[0], len(y_) : len(y_) + dev_y.shape[0]]).dot(F[i][len(y_) : len(y_) + dev_y.shape[0], : ])) \
        #          + lamda * np.sum(np.array(F[i][len(y_) : len(y_) + dev_y.shape[0], :] - Y_dev) ** 2) \
        #         + gama * np.sum(np.array(pred_to_logs[i] - dev_label) ** 2)
        # F_dev = F[i][len(y_): len(y_) + dev_y.shape[0], :]
        # phi[i] = lamda * loss_F_Y(F_dev, Y_dev) \
        #          + gama * np.sum(np.array(pred_to_logs[i] - dev_label)[pred_to_logs[i] == 1] ** 2)
        phi[i] = lamda * np.sum(np.array(F[i][len(y_): len(y_) + dev_y.shape[0], :] - Y_dev)[_pred[i] == 1] ** 2) \
                 + gama * np.sum(np.array(pred_to_logs[i] - dev_label)[pred_to_logs[i] == 1] ** 2)
        # print(lamda * alpha[i] * np.sum(np.array(F[i][len(y_) : len(y_) + dev_y.shape[0], :] - Y_dev) ** 2)
        #     , gama * alpha[i] * np.sum(np.array(pred_to_logs[i] - dev_label) ** 2))
        # + np.trace(F[i].T.dot(L2[i].toarray()).dot(F[i]))
    # print('phi=', phi)
    # phi = 1 / phi
    # phi /= sum(phi)
    print('phi=', phi)
    base = 1 if (num_hg * np.max(phi) - np.sum(phi)) / 2 == 0 else (num_hg * np.max(phi) - np.sum(phi)) / 2
    zeta = np.power(10.0, int(math.log10(base) + 1))
    print(zeta)
    for i in range(num_hg):
        alpha[i] = 1 / num_hg + (np.sum(phi) / (2 * num_hg * zeta) - (phi[i] / (2 * zeta)))
    # min_, max_ = np.argmin(alpha), np.argmax(alpha)
    # tmp = alpha[min_]
    # alpha[min_] = alpha[max_]
    # alpha[max_] = tmp
    # alpha /= sum(alpha)
    print('alpha=', alpha)
    print('--MVHL--')
    Last_F = F
    Last_alpha = alpha
    for iter in range(iters):
        # for i in range(num_hg):
        #     Theta[i] = sqInvDv[i].dot(U[i]).dot(H[i]).dot(W[i]).dot(invDe[i]).dot(H[i].T).dot(U[i]).dot(sqInvDv[i])
        #     L2[i] = U[i] - Theta[i]
        F = Last_F
        for i in range(num_hg):
            V = Y
            # V[y == -1] = F[i][y == -1]
            # V = (V - V.min(axis=1, keepdims=True)) / (V.max(axis=1, keepdims=True) - V.min(axis=1, keepdims=True))
            F[i] = inv(alpha[i] * (L2[i].toarray()) + lamda * alpha[i] * I + mu * I). \
                dot(lamda * alpha[i] * V + mu * cal_F_V(i, Last_F, alpha))
            # row_sums = F[i].sum(axis=1)
            # F[i] /= row_sums

        for i in range(num_hg):
            pre = np.argmax(np.array(F[i]), axis=1)[len_dev:]
            # res.append(pre)
            model_eval(test_y, pre)
            # pre = np.argmax(np.array(F[i]), axis=1)[len(y_): len(y_) + len(dev_y)]
            # print(len(pre), len(dev_y))
            # model_eval(dev_y, pre)
        #     hg[i].update_W(F[i])
        #     sqInvDv[i] = hg[i].sqinvDv
        #     W[i] = hg[i].W
        pred_to_logs = []
        _pred = []
        for i in range(num_hg):
            pred = np.argmax(np.array(F[i]), axis=1)
            pred = pred[len(y_):len_dev]
            _pred.append(pred)
            p_t_l = np.zeros(dev_label.shape[0])
            for j in range(len(pred)):
                p_t_l[id_dev[j]] = pred[j]
            pred_to_logs.append(p_t_l)
        phi = np.zeros(num_hg)
        for i in range(num_hg):
            # phi[i] = lamda * np.sum(np.array(np.argmax(F[i], axis=1) - np.argmax(Y, axis=1))[y != -1] ** 2)
            # phi[i] = lamda * np.sum(np.array(np.argmax(np.array(F[i]), axis=1)[len(y_): len_dev] - dev_y) ** 2) \
            #          + gama * np.sum(np.array(pred_to_logs[i] - dev_label) ** 2)
            # phi[i] = 1 / np.trace(F[i][len(y_): len(y_) + dev_y.shape[0], :].T.dot(
            #     L2[i].toarray()[len(y_): len(y_) + dev_y.shape[0], len(y_): len(y_) + dev_y.shape[0]]).dot(
            #     F[i][len(y_): len(y_) + dev_y.shape[0], :])) \
            #          + lamda * np.sum(np.array(F[i][len(y_): len(y_) + dev_y.shape[0], :] - Y_dev) ** 2) \
            #          + gama * np.sum(np.array(pred_to_logs[i] - dev_label) ** 2)
            # F_dev = F[i][len(y_): len(y_) + dev_y.shape[0], :]
            # phi[i] = lamda * loss_F_Y(F_dev, Y_dev) \
            #          + gama * np.sum(np.array(pred_to_logs[i] - dev_label)[pred_to_logs[i] == 1] ** 2)
            phi[i] = lamda * np.sum(np.array(F[i][len(y_): len(y_) + dev_y.shape[0], :] - Y_dev)[_pred[i] == 1] ** 2) \
                     + gama * np.sum(np.array(pred_to_logs[i] - dev_label)[pred_to_logs[i] == 1] ** 2)
            # print(lamda * np.sum(np.array(F[i][len(y_): len(y_) + dev_y.shape[0], :] - Y_dev) ** 2)
            #       , gama * np.sum(np.array(pred_to_logs[i] - dev_label) ** 2))
            # print(lamda * np.sum(np.array(F[i][len(y_): len(y_) + dev_y.shape[0], :] - Y_dev)[_pred[i] == 1] ** 2)
            #         , gama * np.sum(np.array(pred_to_logs[i] - dev_label)[pred_to_logs[i] == 1] ** 2), gama * np.sum(np.array(pred_to_logs[i] - dev_label)[pred_to_logs[i] == 0] ** 2))
        print('phi=', phi)
        # phi = 1 / phi
        # phi /= sum(phi)
        base = 1 if (num_hg * np.max(phi) - np.sum(phi)) / 2 == 0 else (num_hg * np.max(phi) - np.sum(phi)) / 2
        print(base, int(math.log10(base) + 1))
        zeta = np.power(10.0, int(math.log10(base) + 1))
        print(zeta)
        for i in range(num_hg):
            alpha[i] = 1 / num_hg + (np.sum(phi) / (2 * num_hg * zeta) - (phi[i] / (2 * zeta)))
        # alpha /= sum(alpha)
        print('alpha=', alpha)
        _loss = zeta * np.sum(alpha ** 2)
        # + np.trace(W[i].toarray().T.dot(W[i].toarray()))
        for i in range(num_hg):
            _loss += mu * cal_DF(i, F, F[i]) + lamda * alpha[i] * np.sum(np.array(F[i][len(y_): len(y_) + dev_y.shape[0], :] - Y_dev) ** 2) \
                     + gama * alpha[i] * np.sum(np.array(dev_label - np.array(pred_to_logs[i])) ** 2)
                # + alpha[i] * np.trace(np.array(F[i]).T.dot(L2[i].toarray()).dot(np.array(F[i])))

        # print(new_value, object_value)
        F_ret = F[0] * alpha[0]
        for i in range(1, num_hg):
            F_ret += F[i] * alpha[i]
        pre = np.argmax(np.array(F_ret), axis=1)[len_dev:]
        reg = model_eval(pre, test_y)
        print('iter: {}, Accuracy={}, Precision={}, Recall = {}, F1={}'.format(iter, reg['ACR'], reg['PRE'], reg['PD'], reg['F1']))
        if loss == -1 or _loss < loss:
            loss = _loss
            Last_F = F
            Last_alpha = alpha
        else:
            break
    F = Last_F
    alpha = Last_alpha
    F_ret = F[0] * alpha[0]
    for i in range(1, num_hg):
        F_ret += F[i] * alpha[i]
    # print(F_ret[y_.shape[0]:])
    print('joint')
    pred = np.argmax(np.array(F_ret), axis=1)[len_dev:]
    model_eval(test_y, pred)
    return pred, res

def loss_F_Y(F, Y):
    diff_F_Y = (F[:, 0] - F[:, 1]) * (Y[:, 0] - Y[:, 1])
    return np.sum(np.where(diff_F_Y < 0, diff_F_Y, 0)) * -1