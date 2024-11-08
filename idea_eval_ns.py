import os.path
import re

import numpy as np
import pandas as pd

from process import node_embedding
from main_eval import joint_dev
from CONSTANTS import *
from Drain.demo import parse
import csv
import time
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
    print('Acc={}, PRE={}, Recall={}, F1={}'.format(out['ACR'], out['PRE'], out['PD'], out['F1']))
    # out['DI'] = round((TP / (TP + FN)) / (TN / (TN + FP)) * 100, 4)
    # out['TPR'] = round(TP / (TP + FN) * 100, 4)
    # out['TNR'] = round(TN / (FP + TN) * 100, 4)
    return out

def replace_nth(text, u, v, replacement):
    replacement = replacement.split(' ')
    # matches = list(re.finditer(r'<\*>', text))
    p = 0
    for i in range(v - u):
        matches = list(re.finditer(r'<\*>', text))
        # print(text, len(matches))
        match = matches[u]  # 第 n 个匹配
        start, end = match.span()
        text = text[:start] + replacement[p] + text[end:]
        p += 1
    return text

def get_data_new(file, dataset):
    if not os.path.exists(os.path.join(PROJECT_ROOT, 'dataset', dataset)):
        # shutil.rmtree(os.path.join(PROJECT_ROOT, 'dataset', dataset))
        os.makedirs(os.path.join(PROJECT_ROOT, 'dataset', dataset))
        # continue
    # else:
    #     os.makedirs(os.path.join(PROJECT_ROOT, 'dataset', dataset))
    # train_file = os.path.join(PROJECT_ROOT, 'dataset', dataset, dataset + '_train.log')
    # test_file = os.path.join(PROJECT_ROOT, 'dataset', dataset, dataset + '_test.log')
    if not os.path.isfile(os.path.join(PROJECT_ROOT, 'dataset', dataset, dataset + '.log_structured.csv')):
        parse(file, dataset, '', os.path.join(PROJECT_ROOT, 'dataset', dataset))
    data = pd.read_csv(os.path.join(PROJECT_ROOT, 'dataset', dataset, dataset + '.log_structured.csv'))
    normal_samples = data[data['Label'] == '-']
    anomaly_samples = data[data['Label'] != '-']
    normal_samples = normal_samples.sample(frac=1).reset_index(drop=True)
    anomaly_samples = anomaly_samples.sample(frac=1).reset_index(drop=True)
    len_n_train = int(0.6 * len(normal_samples))
    len_a_train = int(0.6 * len(anomaly_samples))
    len_n_dev = int(0.1 * len(normal_samples))
    len_a_dev = int(0.1 * len(anomaly_samples))
    train_logs = pd.concat([normal_samples[:len_n_train], anomaly_samples[:len_a_train]])
    dev_logs = pd.concat(([normal_samples[len_n_train : len_n_train + len_n_dev], anomaly_samples[len_a_train : len_a_train + len_a_dev]]))
    test_logs = pd.concat([normal_samples[len_n_train + len_n_dev:], anomaly_samples[len_a_train + len_a_dev:]])
    test_logs_labels  = np.concatenate([np.zeros(len(normal_samples) - (len_n_train + len_n_dev)), np.ones(len(anomaly_samples) - (len_a_train + len_a_dev))])
    train_logs_labels = np.concatenate([np.zeros(len_n_train), np.ones(len_a_train)])
    dev_logs_labels = np.concatenate([np.zeros(len_n_dev), np.ones(len_a_dev)])
    train_nodes = train_logs['EventTemplate'].unique()
    test_nodes = test_logs['EventTemplate'].unique()
    dev_nodes = dev_logs['EventTemplate'].unique()
    train_nodes = np.array(train_nodes)
    test_nodes = np.array(test_nodes)
    dev_nodes = np.array(dev_nodes)
    train_labels = np.zeros(train_nodes.shape[0], dtype=int)
    test_labels = np.zeros(test_nodes.shape[0], dtype=int)
    dev_labels = np.zeros(dev_nodes.shape[0], dtype=int)
    for i, node in enumerate(train_nodes):
        l_ = train_logs[train_logs['EventTemplate'] == node]['Label']
        y_ = 0
        for _ in l_:
            y_ += 0 if _ == '-' else 1
        train_labels[i] = 1 if y_ > 0 else 0
    for i, node in enumerate(dev_nodes):
        l_ = dev_logs[dev_logs['EventTemplate'] == node]['Label']
        y_ = 0
        for _ in l_:
            y_ += 0 if _ == '-' else 1
        dev_labels[i] = 1 if y_ > 0 else 0
    for i, node in enumerate(test_nodes):
        l_ = test_logs[test_logs['EventTemplate'] == node]['Label']
        y_ = 0
        for _ in l_:
            y_ += 0 if _ == '-' else 1
        test_labels[i] = 1 if y_ > 0 else 0
    id_train = {}
    for i, node in enumerate(train_nodes):
        id_train[i] = np.where(train_logs['EventTemplate'] == node)[0]
    id_dev = {}
    for i, node in enumerate(dev_nodes):
        id_dev[i] = np.where(dev_logs['EventTemplate'] == node)[0]
    id_test = {}
    for i, node in enumerate(test_nodes):
        id_test[i] = np.where(test_logs['EventTemplate'] == node)[0]
    u = np.zeros(len(train_nodes))
    for i in range(len(train_nodes)):
        u[i] = len(id_train[i])
    sumn = np.sum(u[train_labels == 0])
    suma = np.sum(u[train_labels == 1])
    u[train_labels == 0] /= sumn
    u[train_labels == 1] /= suma
    # for node in test_nodes:
    #     if node not in train_nodes:
    #         print(node)
    return np.concatenate([train_nodes, dev_nodes, test_nodes]), \
        np.concatenate([train_labels, dev_labels, test_labels]), \
        train_logs_labels, test_logs_labels, dev_logs_labels, test_nodes, len(train_nodes), len(train_nodes) + len(dev_nodes), u, id_train, id_test, id_dev

def calculate_variance(data):
    n = len(data)
    if n == 0:
        return 0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    return variance
if __name__ == '__main__':
    for file in ['Spirit.log']:
        # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
        # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
        # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
        # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
        HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
        VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
        JHL_ACC, JHL_PRE, JHL_PF, JHL_PD, JHL_F1 = [], [], [], [], []
        J_VHL_ACC, J_VHL_PRE, J_VHL_PF, J_VHL_PD, J_VHL_F1 = [], [], [], [], []
        time1 = time.time()
        dataset = file.split('.')[0]
        # nodes, label, train_label, test_label, nodes_test, index, u, id_train, id_test = get_data_new(file, dataset)
        # w2v_data, tf_data, bert_data, glove_data = node_embedding(nodes)
        # for f in hl:
        #     model_eval(f, label[index:])
        # for f in vhl:
        #     model_eval(f, label[index:])
        # model_eval(pred_jhl, label[index:])
        # model_eval(pred_jvhl, label[index:])
        sum = 0
        num_hg = 3
        for iter in range(10):
            # with open('data/spirit2', 'r') as file:
            #     lines = file.readlines()
            dataset = file.split('.')[0]
            nodes, label, train_label, test_label, dev_label, nodes_test, len_t, len_tAd, u, id_train, id_test, id_dev = get_data_new(file, dataset)
            print(len(nodes), len(label[label == 1]), len(label[label == 0]))
            embeddings = node_embedding(nodes, 4)
            sum += len(nodes)
        print(sum / 10)
            # pred_jhl, hl = HL(embeddings, label[:index], label[index:], train_label, id_train, 1, 1, 1, 1)
        #     pred_jvhl, vhl = joint_dev(embeddings, label[:len_t], label[len_tAd:], label[len_t : len_tAd], dev_label, id_dev, u, len_tAd, lamda=1, mu=1, zeta=1, gama=1)
        #     # model_eval(label[index:], pred_jhl)
        #     model_eval(label[len_tAd:], pred_jvhl)
        #     # y_jhl = np.ones(len(test_label)) * -1
        #     y_jvhl = np.ones(len(test_label)) * -1
        #     # y_hl = np.zeros((num_hg, len(test_label)))
        #     y_vhl = np.zeros((num_hg, len(test_label)))
        #     for i in range(len(nodes_test)):
        #         # y_jhl[id_test[i]] = pred_jhl[i]
        #         y_jvhl[id_test[i]] = pred_jvhl[i]
        #         for j in range(num_hg):
        #             # y_hl[j, id_test[i]] = hl[j][i]
        #             y_vhl[j, id_test[i]] = vhl[j][i]
        #     # print('-----HL-----')
        #     # for i in range(num_hg):
        #     #     reg = model_eval(test_label, y_hl[i])
        #     #     HL_ACC[i].append(reg['ACR'])
        #     #     HL_PRE[i].append(reg['PRE'])
        #     #     HL_PF[i].append(reg['PF'])
        #     #     HL_PD[i].append(reg['PD'])
        #     #     HL_F1[i].append(reg['F1'])
        #     print('-----VHL-----')
        #     for i in range(num_hg):
        #         reg = model_eval(test_label, y_vhl[i])
        #         VHL_ACC[i].append(reg['ACR'])
        #         VHL_PRE[i].append(reg['PRE'])
        #         VHL_PF[i].append(reg['PF'])
        #         VHL_PD[i].append(reg['PD'])
        #         VHL_F1[i].append(reg['F1'])
        #     # print('-----MHL-----')
        #     # logisitic_reg = model_eval(test_label, y_jhl)
        #     # JHL_ACC.append(logisitic_reg['ACR'])
        #     # JHL_PRE.append(logisitic_reg['PRE'])
        #     # JHL_PF.append(logisitic_reg['PF'])
        #     # JHL_PD.append(logisitic_reg['PD'])
        #     # JHL_F1.append(logisitic_reg['F1'])
        #     print('-----MVHL-----')
        #     logisitic_reg = model_eval(test_label, y_jvhl)
        #     J_VHL_ACC.append(logisitic_reg['ACR'])
        #     J_VHL_PRE.append(logisitic_reg['PRE'])
        #     J_VHL_PF.append(logisitic_reg['PF'])
        #     J_VHL_PD.append(logisitic_reg['PD'])
        #     J_VHL_F1.append(logisitic_reg['F1'])
        # result = 'record/_idea_ns.csv'
        # with open(result, 'a+', newline='') as file:
        #     writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
        #     for i in range(num_hg):
        #         res = []
        #         res.append(f'{round(np.average(HL_ACC[i]), 4)}({round(calculate_variance(HL_ACC[i]), 4)})')
        #         res.append(f'{round(np.average(HL_PRE[i]), 4)}({round(calculate_variance(HL_PRE[i]), 4)})')
        #         res.append(f'{round(np.average(HL_PF[i]), 4)}({round(calculate_variance(HL_PF[i]), 4)})')
        #         res.append(f'{round(np.average(HL_PD[i]), 4)}({round(calculate_variance(HL_PD[i]), 4)})')
        #         res.append(f'{round(np.average(HL_F1[i]), 4)}({round(calculate_variance(HL_F1[i]), 4)})')
        #         writer.writerow(res)
        #     for i in range(num_hg):
        #         res = []
        #         res.append(f'{round(np.average(VHL_ACC[i]), 4)}({round(calculate_variance(VHL_ACC[i]), 4)})')
        #         res.append(f'{round(np.average(VHL_PRE[i]), 4)}({round(calculate_variance(VHL_PRE[i]), 4)})')
        #         res.append(f'{round(np.average(VHL_PF[i]), 4)}({round(calculate_variance(VHL_PF[i]), 4)})')
        #         res.append(f'{round(np.average(VHL_PD[i]), 4)}({round(calculate_variance(VHL_PD[i]), 4)})')
        #         res.append(f'{round(np.average(VHL_F1[i]), 4)}({round(calculate_variance(VHL_F1[i]), 4)})')
        #         writer.writerow(res)
        #     res = []
        #     res.append(f'{round(np.average(JHL_ACC), 4)}({round(calculate_variance(JHL_ACC), 4)})')
        #     res.append(f'{round(np.average(JHL_PRE), 4)}({round(calculate_variance(JHL_PRE), 4)})')
        #     res.append(f'{round(np.average(JHL_PF), 4)}({round(calculate_variance(JHL_PF), 4)})')
        #     res.append(f'{round(np.average(JHL_PD), 4)}({round(calculate_variance(JHL_PD), 4)})')
        #     res.append(f'{round(np.average(JHL_F1), 4)}({round(calculate_variance(JHL_F1), 4)})')
        #     writer.writerow(res)
        #     res = []
        #     res.append(f'{round(np.average(J_VHL_ACC), 4)}({round(calculate_variance(J_VHL_ACC), 4)})')
        #     res.append(f'{round(np.average(J_VHL_PRE), 4)}({round(calculate_variance(J_VHL_PRE), 4)})')
        #     res.append(f'{round(np.average(J_VHL_PF), 4)}({round(calculate_variance(J_VHL_PF), 4)})')
        #     res.append(f'{round(np.average(J_VHL_PD), 4)}({round(calculate_variance(J_VHL_PD), 4)})')
        #     res.append(f'{round(np.average(J_VHL_F1), 4)}({round(calculate_variance(J_VHL_F1), 4)})')
        #     writer.writerow(res)

# if __name__ == '__main__':
#     for file in ['BGL_2k.log'] :
#         for lamda in [1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
#             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
#             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
#             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
#             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
#             HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
#             VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
#             JHL_ACC, JHL_PRE, JHL_PF, JHL_PD, JHL_F1 = [], [], [], [], []
#             J_VHL_ACC, J_VHL_PRE, J_VHL_PF, J_VHL_PD, J_VHL_F1 = [], [], [], [], []
#             time1 = time.time()
#             dataset = str(file).split('.')[0]
#             # nodes, label, train_label, test_label, nodes_test, index, u, id_train, id_test = get_data_new(file, dataset)
#             # w2v_data, tf_data, bert_data, glove_data = node_embedding(nodes)
#             # for f in hl:
#             #     model_eval(f, label[index:])
#             # for f in vhl:
#             #     model_eval(f, label[index:])
#             # model_eval(pred_jhl, label[index:])
#             # model_eval(pred_jvhl, label[index:])
#             for iter in range(10):
#                 # with open('data/spirit2', 'r') as file:
#                 #     lines = file.readlines()
#                 dataset = str(file).split('.')[0]
#                 nodes, label, train_label, test_label, dev_label, nodes_test, len_t, len_tAd, u, id_train, id_test, id_dev = get_data_new(
#                     file, dataset)
#                 #             print(len(nodes), len(label[label == 1]), len(label[label == 0]))
#                 embeddings = node_embedding(nodes, 4)
#                 num_hg = 3
#                 dataset = file.split('.')[0]
#                 # pred_jhl, hl = HL(embeddings, label[:index], label[index:], train_label, id_train, 1, 1, 1, 1)
#                 pred_jvhl, vhl = joint_dev(embeddings, label[:len_t], label[len_tAd:], label[len_t : len_tAd], dev_label, id_dev, u, len_tAd, lamda=lamda, mu=1, zeta=1, gama=1)
#                 model_eval(label[len_tAd:], pred_jvhl)
#                 # y_jhl = np.ones(len(test_label)) * -1
#                 y_jvhl = np.ones(len(test_label)) * -1
#                 # y_hl = np.zeros((num_hg, len(test_label)))
#                 y_vhl = np.zeros((num_hg, len(test_label)))
#                 for i in range(len(nodes_test)):
#                     # y_jhl[id_test[i]] = pred_jhl[i]
#                     y_jvhl[id_test[i]] = pred_jvhl[i]
#                     for j in range(num_hg):
#                         # y_hl[j, id_test[i]] = hl[j][i]
#                         y_vhl[j, id_test[i]] = vhl[j][i]
#                 # print('-----HL-----')
#                 # for i in range(num_hg):
#                 #     reg = model_eval(test_label, y_hl[i])
#                 #     HL_ACC[i].append(reg['ACR'])
#                 #     HL_PRE[i].append(reg['PRE'])
#                 #     HL_PF[i].append(reg['PF'])
#                 #     HL_PD[i].append(reg['PD'])
#                 #     HL_F1[i].append(reg['F1'])
#                 # print('-----VHL-----')
#                 # for i in range(num_hg):
#                 #     reg = model_eval(test_label, y_vhl[i])
#                 #     VHL_ACC[i].append(reg['ACR'])
#                 #     VHL_PRE[i].append(reg['PRE'])
#                 #     VHL_PF[i].append(reg['PF'])
#                 #     VHL_PD[i].append(reg['PD'])
#                 #     VHL_F1[i].append(reg['F1'])
#                 # print('-----MHL-----')
#                 # logisitic_reg = model_eval(test_label, y_jhl)
#                 # JHL_ACC.append(logisitic_reg['ACR'])
#                 # JHL_PRE.append(logisitic_reg['PRE'])
#                 # JHL_PF.append(logisitic_reg['PF'])
#                 # JHL_PD.append(logisitic_reg['PD'])
#                 # JHL_F1.append(logisitic_reg['F1'])
#                 print('-----MVHL-----')
#                 logisitic_reg = model_eval(test_label, y_jvhl)
#                 J_VHL_ACC.append(logisitic_reg['ACR'])
#                 J_VHL_PRE.append(logisitic_reg['PRE'])
#                 J_VHL_PF.append(logisitic_reg['PF'])
#                 J_VHL_PD.append(logisitic_reg['PD'])
#                 J_VHL_F1.append(logisitic_reg['F1'])
#             result = 'record/_idea_lamda_.csv'
#             with open(result, 'a+', newline='') as f:
#                 writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 # for i in range(num_hg):
#                 #     res = []
#                 #     res.append(f'{round(np.average(HL_ACC[i]), 4)}({round(calculate_variance(HL_ACC[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_PRE[i]), 4)}({round(calculate_variance(HL_PRE[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_PF[i]), 4)}({round(calculate_variance(HL_PF[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_PD[i]), 4)}({round(calculate_variance(HL_PD[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_F1[i]), 4)}({round(calculate_variance(HL_F1[i]), 4)})')
#                 #     writer.writerow(res)
#                 # for i in range(num_hg):
#                 #     res = []
#                 #     res.append(f'{round(np.average(VHL_ACC[i]), 4)}({round(calculate_variance(VHL_ACC[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_PRE[i]), 4)}({round(calculate_variance(VHL_PRE[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_PF[i]), 4)}({round(calculate_variance(VHL_PF[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_PD[i]), 4)}({round(calculate_variance(VHL_PD[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_F1[i]), 4)}({round(calculate_variance(VHL_F1[i]), 4)})')
#                 #     writer.writerow(res)
#                 # res = []
#                 # res.append(f'{round(np.average(JHL_ACC), 4)}({round(calculate_variance(JHL_ACC), 4)})')
#                 # res.append(f'{round(np.average(JHL_PRE), 4)}({round(calculate_variance(JHL_PRE), 4)})')
#                 # res.append(f'{round(np.average(JHL_PF), 4)}({round(calculate_variance(JHL_PF), 4)})')
#                 # res.append(f'{round(np.average(JHL_PD), 4)}({round(calculate_variance(JHL_PD), 4)})')
#                 # res.append(f'{round(np.average(JHL_F1), 4)}({round(calculate_variance(JHL_F1), 4)})')
#                 # writer.writerow(res)
#                 res = []
#                 res.append(f'{round(np.average(J_VHL_ACC), 4)}({round(calculate_variance(J_VHL_ACC), 4)})')
#                 res.append(f'{round(np.average(J_VHL_PRE), 4)}({round(calculate_variance(J_VHL_PRE), 4)})')
#                 res.append(f'{round(np.average(J_VHL_PF), 4)}({round(calculate_variance(J_VHL_PF), 4)})')
#                 res.append(f'{round(np.average(J_VHL_PD), 4)}({round(calculate_variance(J_VHL_PD), 4)})')
#                 res.append(f'{round(np.average(J_VHL_F1), 4)}({round(calculate_variance(J_VHL_F1), 4)})')
#                 writer.writerow(res)
# if __name__ == '__main__':
#     for file in ['BGL_2k.log'] :
#         for mu in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
#             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
#             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
#             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
#             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
#             HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
#             VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
#             JHL_ACC, JHL_PRE, JHL_PF, JHL_PD, JHL_F1 = [], [], [], [], []
#             J_VHL_ACC, J_VHL_PRE, J_VHL_PF, J_VHL_PD, J_VHL_F1 = [], [], [], [], []
#             time1 = time.time()
#             dataset = file.split('.')[0]
#             # nodes, label, train_label, test_label, nodes_test, index, u, id_train, id_test = get_data_new(file, dataset)
#             # w2v_data, tf_data, bert_data, glove_data = node_embedding(nodes)
#             # for f in hl:
#             #     model_eval(f, label[index:])
#             # for f in vhl:
#             #     model_eval(f, label[index:])
#             # model_eval(pred_jhl, label[index:])
#             # model_eval(pred_jvhl, label[index:])
#             for iter in range(10):
#                 # with open('data/spirit2', 'r') as file:
#                 #     lines = file.readlines()
#                 nodes, label, train_label, test_label, dev_label, nodes_test, len_t, len_tAd, u, id_train, id_test, id_dev = get_data_new(
#                     file, dataset)
#                 #             print(len(nodes), len(label[label == 1]), len(label[label == 0]))
#                 embeddings = node_embedding(nodes, 4)
#                 num_hg = 3
#                 dataset = file.split('.')[0]
#                 # pred_jhl, hl = HL(embeddings, label[:index], label[index:], train_label, id_train, 1, 1, 1, 1)
#                 pred_jvhl, vhl = joint_dev(embeddings, label[:len_t], label[len_tAd:], label[len_t: len_tAd], dev_label,
#                                            id_dev, u, len_tAd, lamda=1, mu=mu, zeta=1, gama=1)
#                 model_eval(label[len_tAd:], pred_jvhl)
#                 # y_jhl = np.ones(len(test_label)) * -1
#                 y_jvhl = np.ones(len(test_label)) * -1
#                 # y_hl = np.zeros((num_hg, len(test_label)))
#                 y_vhl = np.zeros((num_hg, len(test_label)))
#                 for i in range(len(nodes_test)):
#                     # y_jhl[id_test[i]] = pred_jhl[i]
#                     y_jvhl[id_test[i]] = pred_jvhl[i]
#                     for j in range(num_hg):
#                         # y_hl[j, id_test[i]] = hl[j][i]
#                         y_vhl[j, id_test[i]] = vhl[j][i]
#                 # print('-----HL-----')
#                 # for i in range(num_hg):
#                 #     reg = model_eval(test_label, y_hl[i])
#                 #     HL_ACC[i].append(reg['ACR'])
#                 #     HL_PRE[i].append(reg['PRE'])
#                 #     HL_PF[i].append(reg['PF'])
#                 #     HL_PD[i].append(reg['PD'])
#                 #     HL_F1[i].append(reg['F1'])
#                 # print('-----VHL-----')
#                 # for i in range(num_hg):
#                 #     reg = model_eval(test_label, y_vhl[i])
#                 #     VHL_ACC[i].append(reg['ACR'])
#                 #     VHL_PRE[i].append(reg['PRE'])
#                 #     VHL_PF[i].append(reg['PF'])
#                 #     VHL_PD[i].append(reg['PD'])
#                 #     VHL_F1[i].append(reg['F1'])
#                 # print('-----MHL-----')
#                 # logisitic_reg = model_eval(test_label, y_jhl)
#                 # JHL_ACC.append(logisitic_reg['ACR'])
#                 # JHL_PRE.append(logisitic_reg['PRE'])
#                 # JHL_PF.append(logisitic_reg['PF'])
#                 # JHL_PD.append(logisitic_reg['PD'])
#                 # JHL_F1.append(logisitic_reg['F1'])
#                 print('-----MVHL-----')
#                 logisitic_reg = model_eval(test_label, y_jvhl)
#                 J_VHL_ACC.append(logisitic_reg['ACR'])
#                 J_VHL_PRE.append(logisitic_reg['PRE'])
#                 J_VHL_PF.append(logisitic_reg['PF'])
#                 J_VHL_PD.append(logisitic_reg['PD'])
#                 J_VHL_F1.append(logisitic_reg['F1'])
#             result = 'record/_idea_mu_.csv'
#             with open(result, 'a+', newline='') as f:
#                 writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 # for i in range(num_hg):
#                 #     res = []
#                 #     res.append(f'{round(np.average(HL_ACC[i]), 4)}({round(calculate_variance(HL_ACC[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_PRE[i]), 4)}({round(calculate_variance(HL_PRE[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_PF[i]), 4)}({round(calculate_variance(HL_PF[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_PD[i]), 4)}({round(calculate_variance(HL_PD[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_F1[i]), 4)}({round(calculate_variance(HL_F1[i]), 4)})')
#                 #     writer.writerow(res)
#                 # for i in range(num_hg):
#                 #     res = []
#                 #     res.append(f'{round(np.average(VHL_ACC[i]), 4)}({round(calculate_variance(VHL_ACC[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_PRE[i]), 4)}({round(calculate_variance(VHL_PRE[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_PF[i]), 4)}({round(calculate_variance(VHL_PF[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_PD[i]), 4)}({round(calculate_variance(VHL_PD[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_F1[i]), 4)}({round(calculate_variance(VHL_F1[i]), 4)})')
#                 #     writer.writerow(res)
#                 # res = []
#                 # res.append(f'{round(np.average(JHL_ACC), 4)}({round(calculate_variance(JHL_ACC), 4)})')
#                 # res.append(f'{round(np.average(JHL_PRE), 4)}({round(calculate_variance(JHL_PRE), 4)})')
#                 # res.append(f'{round(np.average(JHL_PF), 4)}({round(calculate_variance(JHL_PF), 4)})')
#                 # res.append(f'{round(np.average(JHL_PD), 4)}({round(calculate_variance(JHL_PD), 4)})')
#                 # res.append(f'{round(np.average(JHL_F1), 4)}({round(calculate_variance(JHL_F1), 4)})')
#                 # writer.writerow(res)
#                 res = []
#                 res.append(f'{round(np.average(J_VHL_ACC), 4)}({round(calculate_variance(J_VHL_ACC), 4)})')
#                 res.append(f'{round(np.average(J_VHL_PRE), 4)}({round(calculate_variance(J_VHL_PRE), 4)})')
#                 res.append(f'{round(np.average(J_VHL_PF), 4)}({round(calculate_variance(J_VHL_PF), 4)})')
#                 res.append(f'{round(np.average(J_VHL_PD), 4)}({round(calculate_variance(J_VHL_PD), 4)})')
#                 res.append(f'{round(np.average(J_VHL_F1), 4)}({round(calculate_variance(J_VHL_F1), 4)})')
#                 writer.writerow(res)
# # if __name__ == '__main__':
# #     for file in ['BGL_2k.log'] :
# #         for zeta in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
# #             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
# #             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
# #             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
# #             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
# #             HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
# #             VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
# #             JHL_ACC, JHL_PRE, JHL_PF, JHL_PD, JHL_F1 = [], [], [], [], []
# #             J_VHL_ACC, J_VHL_PRE, J_VHL_PF, J_VHL_PD, J_VHL_F1 = [], [], [], [], []
# #             time1 = time.time()
# #             dataset = file.split('.')[0]
# #             # nodes, label, train_label, test_label, nodes_test, index, u, id_train, id_test = get_data_new(file, dataset)
# #             # w2v_data, tf_data, bert_data, glove_data = node_embedding(nodes)
# #             # for f in hl:
# #             #     model_eval(f, label[index:])
# #             # for f in vhl:
# #             #     model_eval(f, label[index:])
# #             # model_eval(pred_jhl, label[index:])
# #             # model_eval(pred_jvhl, label[index:])
# #             for iter in range(10):
# #                 # with open('data/spirit2', 'r') as file:
# #                 #     lines = file.readlines()
# #                 dataset = file.split('.')[0]
# #                 nodes, label, train_label, test_label, nodes_test, index, u, id_train, id_test = get_data_new(file, dataset)
# #                 print(len(nodes), len(label[label == 1]), len(label[label == 0]))
# #                 embeddings = node_embedding(nodes, 4)
# #                 num_hg = len(embeddings)
# #                 # pred_jhl, hl = HL(embeddings, label[:index], label[index:], train_label, id_train, 1, 1, 1, 10000)
# #                 pred_jvhl, vhl = joint(embeddings, label[:index], label[index:], train_label, id_train, u, lamda=1, mu=1, zeta=zeta, gama=1)
# #                 # model_eval(label[index:], pred_jhl)
# #                 model_eval(label[index:], pred_jvhl)
# #                 # y_jhl = np.zeros(len(test_label))
# #                 y_jvhl = np.zeros(len(test_label))
# #                 # y_hl = np.zeros((num_hg, len(test_label)))
# #                 # y_vhl = np.zeros((num_hg, len(test_label)))
# #                 for i in range(len(nodes_test)):
# #                     # y_jhl[id_test[i]] = pred_jhl[i]
# #                     y_jvhl[id_test[i]] = pred_jvhl[i]
# #                     # for j in range(num_hg):
# #                         # y_hl[j, id_test[i]] = hl[j][i]
# #                         # y_vhl[j, id_test[i]] = vhl[j][i]
# #                 # print('-----HL-----')
# #                 # for i in range(num_hg):
# #                 #     reg = model_eval(test_label, y_hl[i])
# #                 #     HL_ACC[i].append(reg['ACR'])
# #                 #     HL_PRE[i].append(reg['PRE'])
# #                 #     HL_PF[i].append(reg['PF'])
# #                 #     HL_PD[i].append(reg['PD'])
# #                 #     HL_F1[i].append(reg['F1'])
# #                 # print('-----VHL-----')
# #                 # for i in range(num_hg):
# #                 #     reg = model_eval(test_label, y_vhl[i])
# #                 #     VHL_ACC[i].append(reg['ACR'])
# #                 #     VHL_PRE[i].append(reg['PRE'])
# #                 #     VHL_PF[i].append(reg['PF'])
# #                 #     VHL_PD[i].append(reg['PD'])
# #                 #     VHL_F1[i].append(reg['F1'])
# #                 # print('-----MHL-----')
# #                 # logisitic_reg = model_eval(test_label, y_jhl)
# #                 # JHL_ACC.append(logisitic_reg['ACR'])
# #                 # JHL_PRE.append(logisitic_reg['PRE'])
# #                 # JHL_PF.append(logisitic_reg['PF'])
# #                 # JHL_PD.append(logisitic_reg['PD'])
# #                 # JHL_F1.append(logisitic_reg['F1'])
# #                 print('-----MVHL-----')
# #                 logisitic_reg = model_eval(test_label, y_jvhl)
# #                 J_VHL_ACC.append(logisitic_reg['ACR'])
# #                 J_VHL_PRE.append(logisitic_reg['PRE'])
# #                 J_VHL_PF.append(logisitic_reg['PF'])
# #                 J_VHL_PD.append(logisitic_reg['PD'])
# #                 J_VHL_F1.append(logisitic_reg['F1'])
# #             result = 'record/_idea_zeta.csv'
# #             with open(result, 'a+', newline='') as f:
# #                 writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
# #                 # for i in range(num_hg):
# #                 #     res = []
# #                 #     res.append(f'{round(np.average(HL_ACC[i]), 4)}({round(calculate_variance(HL_ACC[i]), 4)})')
# #                 #     res.append(f'{round(np.average(HL_PRE[i]), 4)}({round(calculate_variance(HL_PRE[i]), 4)})')
# #                 #     res.append(f'{round(np.average(HL_PF[i]), 4)}({round(calculate_variance(HL_PF[i]), 4)})')
# #                 #     res.append(f'{round(np.average(HL_PD[i]), 4)}({round(calculate_variance(HL_PD[i]), 4)})')
# #                 #     res.append(f'{round(np.average(HL_F1[i]), 4)}({round(calculate_variance(HL_F1[i]), 4)})')
# #                 #     writer.writerow(res)
# #                 # for i in range(num_hg):
# #                 #     res = []
# #                 #     res.append(f'{round(np.average(VHL_ACC[i]), 4)}({round(calculate_variance(VHL_ACC[i]), 4)})')
# #                 #     res.append(f'{round(np.average(VHL_PRE[i]), 4)}({round(calculate_variance(VHL_PRE[i]), 4)})')
# #                 #     res.append(f'{round(np.average(VHL_PF[i]), 4)}({round(calculate_variance(VHL_PF[i]), 4)})')
# #                 #     res.append(f'{round(np.average(VHL_PD[i]), 4)}({round(calculate_variance(VHL_PD[i]), 4)})')
# #                 #     res.append(f'{round(np.average(VHL_F1[i]), 4)}({round(calculate_variance(VHL_F1[i]), 4)})')
# #                 #     writer.writerow(res)
# #                 # res = []
# #                 # res.append(f'{round(np.average(JHL_ACC), 4)}({round(calculate_variance(JHL_ACC), 4)})')
# #                 # res.append(f'{round(np.average(JHL_PRE), 4)}({round(calculate_variance(JHL_PRE), 4)})')
# #                 # res.append(f'{round(np.average(JHL_PF), 4)}({round(calculate_variance(JHL_PF), 4)})')
# #                 # res.append(f'{round(np.average(JHL_PD), 4)}({round(calculate_variance(JHL_PD), 4)})')
# #                 # res.append(f'{round(np.average(JHL_F1), 4)}({round(calculate_variance(JHL_F1), 4)})')
# #                 # writer.writerow(res)
# #                 res = []
# #                 res.append(f'{round(np.average(J_VHL_ACC), 4)}({round(calculate_variance(J_VHL_ACC), 4)})')
# #                 res.append(f'{round(np.average(J_VHL_PRE), 4)}({round(calculate_variance(J_VHL_PRE), 4)})')
# #                 res.append(f'{round(np.average(J_VHL_PF), 4)}({round(calculate_variance(J_VHL_PF), 4)})')
# #                 res.append(f'{round(np.average(J_VHL_PD), 4)}({round(calculate_variance(J_VHL_PD), 4)})')
# #                 res.append(f'{round(np.average(J_VHL_F1), 4)}({round(calculate_variance(J_VHL_F1), 4)})')
# #                 writer.writerow(res)
# if __name__ == '__main__':
#     for file in ['BGL_2k.log'] :
#         for gama in [1e-4, 1e-3, 1e-2, 1e-1, 1, 1e1, 1e2, 1e3, 1e4]:
#             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
#             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
#             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
#             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
#             HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
#             VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
#             JHL_ACC, JHL_PRE, JHL_PF, JHL_PD, JHL_F1 = [], [], [], [], []
#             J_VHL_ACC, J_VHL_PRE, J_VHL_PF, J_VHL_PD, J_VHL_F1 = [], [], [], [], []
#             time1 = time.time()
#             dataset = file.split('.')[0]
#             # nodes, label, train_label, test_label, nodes_test, index, u, id_train, id_test = get_data_new(file, dataset)
#             # w2v_data, tf_data, bert_data, glove_data = node_embedding(nodes)
#             # for f in hl:
#             #     model_eval(f, label[index:])
#             # for f in vhl:
#             #     model_eval(f, label[index:])
#             # model_eval(pred_jhl, label[index:])
#             # model_eval(pred_jvhl, label[index:])
#             for iter in range(10):
#                 # with open('data/spirit2', 'r') as file:
#                 #     lines = file.readlines()
#                 nodes, label, train_label, test_label, dev_label, nodes_test, len_t, len_tAd, u, id_train, id_test, id_dev = get_data_new(
#                     file, dataset)
#                 #             print(len(nodes), len(label[label == 1]), len(label[label == 0]))
#                 embeddings = node_embedding(nodes, 4)
#                 num_hg = 3
#                 dataset = file.split('.')[0]
#                 # pred_jhl, hl = HL(embeddings, label[:index], label[index:], train_label, id_train, 1, 1, 1, 1)
#                 pred_jvhl, vhl = joint_dev(embeddings, label[:len_t], label[len_tAd:], label[len_t: len_tAd], dev_label,
#                                            id_dev, u, len_tAd, lamda=1, mu=1, zeta=1, gama=gama)
#                 model_eval(label[len_tAd:], pred_jvhl)
#                 # y_jhl = np.ones(len(test_label)) * -1
#                 y_jvhl = np.ones(len(test_label)) * -1
#                 # y_hl = np.zeros((num_hg, len(test_label)))
#                 y_vhl = np.zeros((num_hg, len(test_label)))
#                 for i in range(len(nodes_test)):
#                     # y_jhl[id_test[i]] = pred_jhl[i]
#                     y_jvhl[id_test[i]] = pred_jvhl[i]
#                     for j in range(num_hg):
#                         # y_hl[j, id_test[i]] = hl[j][i]
#                         y_vhl[j, id_test[i]] = vhl[j][i]
#                 # print('-----HL-----')
#                 # for i in range(num_hg):
#                 #     reg = model_eval(test_label, y_hl[i])
#                 #     HL_ACC[i].append(reg['ACR'])
#                 #     HL_PRE[i].append(reg['PRE'])
#                 #     HL_PF[i].append(reg['PF'])
#                 #     HL_PD[i].append(reg['PD'])
#                 #     HL_F1[i].append(reg['F1'])
#                 # print('-----VHL-----')
#                 # for i in range(num_hg):
#                 #     reg = model_eval(test_label, y_vhl[i])
#                 #     VHL_ACC[i].append(reg['ACR'])
#                 #     VHL_PRE[i].append(reg['PRE'])
#                 #     VHL_PF[i].append(reg['PF'])
#                 #     VHL_PD[i].append(reg['PD'])
#                 #     VHL_F1[i].append(reg['F1'])
#                 # print('-----MHL-----')
#                 # logisitic_reg = model_eval(test_label, y_jhl)
#                 # JHL_ACC.append(logisitic_reg['ACR'])
#                 # JHL_PRE.append(logisitic_reg['PRE'])
#                 # JHL_PF.append(logisitic_reg['PF'])
#                 # JHL_PD.append(logisitic_reg['PD'])
#                 # JHL_F1.append(logisitic_reg['F1'])
#                 print('-----MVHL-----')
#                 logisitic_reg = model_eval(test_label, y_jvhl)
#                 J_VHL_ACC.append(logisitic_reg['ACR'])
#                 J_VHL_PRE.append(logisitic_reg['PRE'])
#                 J_VHL_PF.append(logisitic_reg['PF'])
#                 J_VHL_PD.append(logisitic_reg['PD'])
#                 J_VHL_F1.append(logisitic_reg['F1'])
#             result = 'record/_idea_gama_.csv'
#             with open(result, 'a+', newline='') as f:
#                 writer = csv.writer(f, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
#                 # for i in range(num_hg):
#                 #     res = []
#                 #     res.append(f'{round(np.average(HL_ACC[i]), 4)}({round(calculate_variance(HL_ACC[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_PRE[i]), 4)}({round(calculate_variance(HL_PRE[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_PF[i]), 4)}({round(calculate_variance(HL_PF[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_PD[i]), 4)}({round(calculate_variance(HL_PD[i]), 4)})')
#                 #     res.append(f'{round(np.average(HL_F1[i]), 4)}({round(calculate_variance(HL_F1[i]), 4)})')
#                 #     writer.writerow(res)
#                 # for i in range(num_hg):
#                 #     res = []
#                 #     res.append(f'{round(np.average(VHL_ACC[i]), 4)}({round(calculate_variance(VHL_ACC[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_PRE[i]), 4)}({round(calculate_variance(VHL_PRE[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_PF[i]), 4)}({round(calculate_variance(VHL_PF[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_PD[i]), 4)}({round(calculate_variance(VHL_PD[i]), 4)})')
#                 #     res.append(f'{round(np.average(VHL_F1[i]), 4)}({round(calculate_variance(VHL_F1[i]), 4)})')
#                 #     writer.writerow(res)
#                 # res = []
#                 # res.append(f'{round(np.average(JHL_ACC), 4)}({round(calculate_variance(JHL_ACC), 4)})')
#                 # res.append(f'{round(np.average(JHL_PRE), 4)}({round(calculate_variance(JHL_PRE), 4)})')
#                 # res.append(f'{round(np.average(JHL_PF), 4)}({round(calculate_variance(JHL_PF), 4)})')
#                 # res.append(f'{round(np.average(JHL_PD), 4)}({round(calculate_variance(JHL_PD), 4)})')
#                 # res.append(f'{round(np.average(JHL_F1), 4)}({round(calculate_variance(JHL_F1), 4)})')
#                 # writer.writerow(res)
#                 res = []
#                 res.append(f'{round(np.average(J_VHL_ACC), 4)}({round(calculate_variance(J_VHL_ACC), 4)})')
#                 res.append(f'{round(np.average(J_VHL_PRE), 4)}({round(calculate_variance(J_VHL_PRE), 4)})')
#                 res.append(f'{round(np.average(J_VHL_PF), 4)}({round(calculate_variance(J_VHL_PF), 4)})')
#                 res.append(f'{round(np.average(J_VHL_PD), 4)}({round(calculate_variance(J_VHL_PD), 4)})')
#                 res.append(f'{round(np.average(J_VHL_F1), 4)}({round(calculate_variance(J_VHL_F1), 4)})')
#                 writer.writerow(res)