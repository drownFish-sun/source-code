import ast
import os.path
import random
import re

import numpy as np
import pandas as pd

from process import emb
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
    extra_nodes = []
    extra_labels = []
    replace = {}
    for i, node in enumerate(train_nodes):
        l_ = train_logs[train_logs['EventTemplate'] == node]['Label']
        y_ = 0
        for _ in l_:
            y_ += 0 if _ == '-' else 1
        if y_ == 0 or y_ == len(l_):
            extra_labels.append(int(1 if y_ > 0 else 0))
            extra_nodes.append(node)
        if y_ > 0 and len(l_) != y_:
            para_n = train_logs[(train_logs['EventTemplate'] == node) & (train_logs['Label'] == '-')]['ParameterList'].drop_duplicates()
            para_a = train_logs[(train_logs['EventTemplate'] == node) & (train_logs['Label'] != '-')]['ParameterList'].drop_duplicates()
            # replace[node] = True
            para_n = list(para_n)
            para_a = list(para_a)
            # print('para_n=', para_n)
            # print('para_a=', para_a)
            # print(node)
            a, b = 0, 0
            while a < len(para_n) and len(ast.literal_eval(para_n[a])) == 0:
                a += 1
            while b < len(para_a) and len(ast.literal_eval(para_a[b])) == 0:
                b += 1
            if a == len(para_n) or b == len(para_a):
                level_n = train_logs[(train_logs['EventTemplate'] == node) & (train_logs['Label'] == '-')]['Level'].drop_duplicates()
                level_a = train_logs[(train_logs['EventTemplate'] == node) & (train_logs['Label'] != '-')]['Level'].drop_duplicates()
                level_n = list(level_n)
                level_a = list(level_a)
                # print('level_n=', level_n)
                # print('level_a=', level_a)
                flag = 1
                for level in level_a:
                    if level in level_n:
                        flag = 0
                        break
                if flag == 1:
                    # print('normal')
                    for j, level in enumerate(level_n):
                        node_ =  node + ' ' + level
                        indexer = (train_logs['EventTemplate'] == node) & (train_logs['Label'] == '-') & (
                                train_logs['Level'] == level)
                        train_logs.loc[indexer, 'EventTemplate'] = node_
                        # if j == 0:
                        #     train_nodes[i] = node_
                        #     train_labels[i] = int(0)
                        # else:
                        if node_ not in train_nodes:
                            extra_nodes.append(node_)
                            extra_labels.append(int(0))
                    # print('abnormal')
                    for level in level_a:
                        node_ =  node + ' ' + level
                        indexer = (train_logs['EventTemplate'] == node) & (train_logs['Label'] != '-') & (
                                train_logs['Level'] == level)
                        train_logs.loc[indexer, 'EventTemplate'] = node_
                        if node_ not in train_nodes:
                            extra_nodes.append(node_)
                            extra_labels.append(int(1))
                replace[node] = -1
                # train_labels[i] = int(1)
                continue

            n_, a_ = ast.literal_eval(para_n[a]), ast.literal_eval(para_a[b])
            k = 0
            while(k < len(n_) and k < len(a_) and (n_[len(n_) - k - 1] == a_[len(a_) - k - 1])):
                k += 1
            replace[node] = k
            # print('normal')
            for j, para in enumerate(para_n):
                actual_list = ast.literal_eval(para)
                node_ = node
                u_ = 0
                for jj in range(len(actual_list) - k - 1):
                    u_ += len(actual_list[jj].split(' '))
                v_ = u_ + len(actual_list[len(actual_list) - k - 1].split(' '))
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = replace_nth(node_, u_, v_, rep)
                indexer = (train_logs['EventTemplate'] == node) & (train_logs['Label'] == '-') & (
                        train_logs['ParameterList'] == para)
                train_logs.loc[indexer, 'EventTemplate'] = node_
                # if j == 0:
                #     train_nodes[i] = node_
                #     train_labels[i] = int(0)
                # else:
                if node_ not in train_nodes:
                    extra_nodes.append(node_)
                    extra_labels.append(int(0))
                # print(node_)
            # print('anomalous')
            for para in para_a:
                actual_list = ast.literal_eval(para)
                node_ = node
                u_ = 0
                for jj in range(len(actual_list) - k - 1):
                    u_ += len(actual_list[jj].split(' '))
                v_ = u_ + len(actual_list[len(actual_list) - k - 1].split(' '))
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = replace_nth(node_, u_, v_, rep)
                indexer = (train_logs['EventTemplate'] == node) & (train_logs['Label'] != '-') & (
                        train_logs['ParameterList'] == para)
                train_logs.loc[indexer, 'EventTemplate'] = node_
                if node_ not in train_nodes:
                    extra_nodes.append(node_)
                    extra_labels.append(int(1))
                # print(node_)
        # if y_ > 0 and len(l_) != y_:
        #     print(node)
        #     print(l_)
    extra_nodes_ = []
    extra_labels_ = []
    for i, extra_node in enumerate(extra_nodes):
        if extra_node not in extra_nodes_:
            extra_nodes_.append(extra_node)
            extra_labels_.append(extra_labels[i])
    train_nodes = np.array(extra_nodes_)
    train_labels = np.array(extra_labels_, dtype=int)

    extra_nodes = []
    extra_labels = []
    for i, node in enumerate(dev_nodes):
        l_ = dev_logs[dev_logs['EventTemplate'] == node]['Label']
        y_ = 0
        for _ in l_:
            y_ += 0 if _ == '-' else 1
        if node not in replace.keys():
            extra_labels.append(int(1 if y_ > 0 else 0))
            extra_nodes.append(node)
        else:
            # print(node)
            para_ = dev_logs[dev_logs['EventTemplate'] == node]['ParameterList'].drop_duplicates()
            para_ = list(para_)
            if replace[node] == -1:
                level_ = dev_logs[dev_logs['EventTemplate'] == node]['Level'].drop_duplicates()
                level_ = list(level_)
                for j, level in enumerate(level_):
                    node_ = node + ' ' + level
                    indexer = (dev_logs['EventTemplate'] == node) & (dev_logs['Level'] == level)
                    _y = list(dev_logs[(dev_logs['EventTemplate'] == node) & (dev_logs['Level'] == level)][
                                  'Label'].drop_duplicates())
                    dev_logs.loc[indexer, 'EventTemplate'] = node_
                    # print(_y)
                    _ = 0
                    if len(_y) == 1:
                        _ = 1 if '-' not in _y else 0
                    else:
                        _ = 1
                    # print('normal' if int(_) == 0 else 'abnormal', node_)
                    # if j == 0:
                    #     test_logs[i] = node_
                    #     test_logs[i] = int(_)
                    # else:
                    if node_ not in dev_nodes:
                        extra_nodes.append(node_)
                        extra_labels.append(int(_))
                # test_labels[i] = int(1)
                continue
            k = replace[node]
            # print(node)
            for j, para in enumerate(para_):
                actual_list = ast.literal_eval(para)
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = node
                u_ = 0
                for jj in range(len(actual_list) - k - 1):
                    u_ += len(actual_list[jj].split(' '))
                v_ = u_ + len(actual_list[len(actual_list) - k - 1].split(' '))
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = replace_nth(node_, u_, v_, rep)
                _y = list(dev_logs[(dev_logs['EventTemplate'] == node) & (dev_logs['ParameterList'] == para)][
                              'Label'].drop_duplicates())
                indexer = (dev_logs['EventTemplate'] == node) & (dev_logs['ParameterList'] == para)
                dev_logs.loc[indexer, 'EventTemplate'] = node_
                # print(_y)
                _ = 0
                if len(_y) == 1:
                    _ = 1 if '-' not in _y else 0
                else:
                    _ = 1
                # print('normal' if int(_) == 0 else 'abnormal', node_)
                # if j == 0:
                #     test_logs[i] = int(_)
                #     test_nodes[i] = node_
                # else:
                if node_ not in dev_nodes:
                    extra_nodes.append(node_)
                    extra_labels.append(int(_))
                # print(node_)
    extra_nodes_ = []
    extra_labels_ = []
    for i, extra_node in enumerate(extra_nodes):
        if extra_node not in extra_nodes_:
            extra_nodes_.append(extra_node)
            extra_labels_.append(extra_labels[i])

    dev_nodes = np.array(extra_nodes_)
    dev_labels = np.array(extra_labels_, dtype=int)

    extra_nodes = []
    extra_labels = []
    for i, node in enumerate(test_nodes):
        l_ = test_logs[test_logs['EventTemplate'] == node]['Label']
        y_ = 0
        for _ in l_:
            y_ += 0 if _ == '-' else 1
        if node not in replace.keys():
            extra_labels.append(int(1 if y_ > 0 else 0))
            extra_nodes.append(node)
        else:
            # print(node)
            para_ = test_logs[test_logs['EventTemplate'] == node]['ParameterList'].drop_duplicates()
            para_ = list(para_)
            if replace[node] == -1:
                level_ = test_logs[test_logs['EventTemplate'] == node]['Level'].drop_duplicates()
                level_ = list(level_)
                for j, level in enumerate(level_):
                    node_ =  node + ' ' + level
                    indexer = (test_logs['EventTemplate'] == node) & (test_logs['Level'] == level)
                    _y = list(test_logs[(test_logs['EventTemplate'] == node) & (test_logs['Level'] == level)]['Label'].drop_duplicates())
                    test_logs.loc[indexer, 'EventTemplate'] = node_
                    # print(_y)
                    _ = 0
                    if len(_y) == 1:
                        _ = 1 if '-' not in _y else 0
                    else:
                        _ = 1
                    # print('normal' if int(_) == 0 else 'abnormal', node_)
                    # if j == 0:
                    #     test_logs[i] = node_
                    #     test_logs[i] = int(_)
                    # else:
                    if node_ not in test_nodes:
                        extra_nodes.append(node_)
                        extra_labels.append(int(_))
                # test_labels[i] = int(1)
                continue
            k = replace[node]
            # print(node)
            for j, para in enumerate(para_):
                actual_list = ast.literal_eval(para)
                u_ = 0
                node_ = node
                for jj in range(len(actual_list) - k - 1):
                    u_ += len(actual_list[jj].split(' '))
                v_ = u_ + len(actual_list[len(actual_list) - k - 1].split(' '))
                rep = actual_list[len(actual_list) - k - 1] if k < len(actual_list) else '<*>'
                node_ = replace_nth(node_, u_, v_, rep)
                _y = list(test_logs[(test_logs['EventTemplate'] == node) & (test_logs['ParameterList'] == para)]['Label'].drop_duplicates())
                indexer = (test_logs['EventTemplate'] == node) & (test_logs['ParameterList'] == para)
                test_logs.loc[indexer, 'EventTemplate'] = node_
                # print(_y)
                _ = 0
                if len(_y) == 1:
                    _ = 1 if '-' not in _y else 0
                else:
                    _ = 1
                # print('normal' if int(_) == 0 else 'abnormal', node_)
                # if j == 0:
                #     test_logs[i] = int(_)
                #     test_nodes[i] = node_
                # else:
                if node_ not in test_nodes:
                    extra_nodes.append(node_)
                    extra_labels.append(int(_))
                # print(node_)
    extra_nodes_ = []
    extra_labels_ = []
    for i, extra_node in enumerate(extra_nodes):
        if extra_node not in extra_nodes_:
            extra_nodes_.append(extra_node)
            extra_labels_.append(extra_labels[i])
    test_nodes = np.array(extra_nodes_)
    test_labels = np.array(extra_labels_, dtype=int)
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
def process_HDFS(file):
    data = np.load(file, allow_pickle=True)
    x_data = data['x_data']
    y_data = data['y_data']
    x_data_0 = x_data[y_data == 0]
    x_data_1 = x_data[y_data == 1]
    random.seed(987654321)
    idx = [i for i in range(x_data_0.shape[0])]
    random.shuffle(idx)
    x_data_0 = x_data_0[idx]
    idx = [i for i in range(x_data_1.shape[0])]
    random.shuffle(idx)
    x_data_1 = x_data_1[idx]

    len_train_0 = int(len(x_data_0) * 0.6)
    len_train_1 = int(len(x_data_1) * 0.6)
    len_dev_0 = int(len(x_data_0) * 0.1)
    len_dev_1 = int(len(x_data_1) * 0.1)
    test_X = np.concatenate([x_data_0[len_train_0 + len_dev_0:], x_data_1[len_train_1 + len_dev_1:]])
    test_Y = np.concatenate([np.zeros(len(x_data_0) - len_train_0 - len_dev_0), np.ones(len(x_data_1) - len_train_1 - len_dev_1)])
    train_X = np.concatenate([x_data_0[: len_train_0], x_data_1[: len_train_1]])
    train_Y = np.concatenate([np.zeros(len_train_0), np.ones(len_train_1)])
    dev_X = np.concatenate([x_data_0[len_train_0 : len_train_0 + len_dev_0], x_data_1[len_train_1 : len_train_1 + len_dev_1]])
    dev_Y = np.concatenate([np.zeros(len_dev_0), np.ones(len_dev_1)])
    train_nodes = []
    for train in train_X:
        train_nodes.append(' '.join(train))
    dev_nodes = []
    for dev in dev_X:
        dev_nodes.append(' '.join(dev))
    test_nodes = []
    for test in test_X:
        test_nodes.append(' '.join(test))
    print(dev_nodes)
    # train_labels = np.array(train_Y, dtype=int)
    # dev_labels =
    # return np.concatenate([train_nodes, dev_nodes, test_nodes]), \
    #     np.concatenate([train_labels, dev_labels, test_labels]), \
    #     train_logs_labels, test_logs_labels, dev_logs_labels, test_nodes, len(train_nodes), len(train_nodes) + len(
    #     dev_nodes), u, id_train, id_test, id_dev
def calculate_variance(data):
    n = len(data)
    if n == 0:
        return 0
    mean = sum(data) / n
    variance = sum((x - mean) ** 2 for x in data) / n
    return variance
if __name__ == '__main__':
    # for file in ['dataset/HDFS/HDFS.npz']:
    #     process_HDFS(file)
    emb = emb()
    for file in ['BGL.log', 'Spirit.log', 'Liberty.log']:
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
        # 1100.6 1587.1 1601.2
        num_hg = 3
        # sum = 0
        for iter in range(1):
            # with open('data/spirit2', 'r') as file:
            #     lines = file.readlines()
            dataset = file.split('.')[0]
            nodes, label, train_label, test_label, dev_label, nodes_test, len_t, len_tAd, u, id_train, id_test, id_dev = get_data_new(file, dataset)
            print(len(nodes), len(label[label == 1]), len(label[label == 0]))
            # sum += len(nodes)
        # print(sum / 10)
            embeddings = emb.node_embedding(nodes, 4)
            # pred_jhl, hl = HL(embeddings, label[:index], label[index:], train_label, id_train, 1, 1, 1, 1)
            pred_jvhl, vhl = joint_dev(embeddings, label[:len_t], label[len_tAd:], label[len_t : len_tAd], dev_label, id_dev, u, len_tAd, lamda=1, mu=1, gama=1)
            # model_eval(label[index:], pred_jhl)
            model_eval(label[len_tAd:], pred_jvhl)
            # y_jhl = np.ones(len(test_label)) * -1
            y_jvhl = np.ones(len(test_label)) * -1
            # y_hl = np.zeros((num_hg, len(test_label)))
            y_vhl = np.zeros((num_hg, len(test_label)))
            for i in range(len(nodes_test)):
                # y_jhl[id_test[i]] = pred_jhl[i]
                y_jvhl[id_test[i]] = pred_jvhl[i]
                for j in range(num_hg):
                    # y_hl[j, id_test[i]] = hl[j][i]
                    y_vhl[j, id_test[i]] = vhl[j][i]
            # print('-----HL-----')
            # for i in range(num_hg):
            #     reg = model_eval(test_label, y_hl[i])
            #     HL_ACC[i].append(reg['ACR'])
            #     HL_PRE[i].append(reg['PRE'])
            #     HL_PF[i].append(reg['PF'])
            #     HL_PD[i].append(reg['PD'])
            #     HL_F1[i].append(reg['F1'])
            print('-----VHL-----')
            for i in range(num_hg):
                reg = model_eval(test_label, y_vhl[i])
                VHL_ACC[i].append(reg['ACR'])
                VHL_PRE[i].append(reg['PRE'])
                VHL_PF[i].append(reg['PF'])
                VHL_PD[i].append(reg['PD'])
                VHL_F1[i].append(reg['F1'])
            # print('-----MHL-----')
            # logisitic_reg = model_eval(test_label, y_jhl)
            # JHL_ACC.append(logisitic_reg['ACR'])
            # JHL_PRE.append(logisitic_reg['PRE'])
            # JHL_PF.append(logisitic_reg['PF'])
            # JHL_PD.append(logisitic_reg['PD'])
            # JHL_F1.append(logisitic_reg['F1'])
            print('-----MVHL-----')
            logisitic_reg = model_eval(test_label, y_jvhl)
            J_VHL_ACC.append(logisitic_reg['ACR'])
            J_VHL_PRE.append(logisitic_reg['PRE'])
            J_VHL_PF.append(logisitic_reg['PF'])
            J_VHL_PD.append(logisitic_reg['PD'])
            J_VHL_F1.append(logisitic_reg['F1'])
        result = 'record/_idea_try.csv'
        with open(result, 'a+', newline='') as file:
            writer = csv.writer(file, delimiter=',', quotechar='"', quoting=csv.QUOTE_MINIMAL)
            for i in range(num_hg):
                res = []
                res.append(f'{round(np.average(HL_ACC[i]), 4)}({round(calculate_variance(HL_ACC[i]), 4)})')
                res.append(f'{round(np.average(HL_PRE[i]), 4)}({round(calculate_variance(HL_PRE[i]), 4)})')
                res.append(f'{round(np.average(HL_PF[i]), 4)}({round(calculate_variance(HL_PF[i]), 4)})')
                res.append(f'{round(np.average(HL_PD[i]), 4)}({round(calculate_variance(HL_PD[i]), 4)})')
                res.append(f'{round(np.average(HL_F1[i]), 4)}({round(calculate_variance(HL_F1[i]), 4)})')
                writer.writerow(res)
            for i in range(num_hg):
                res = []
                res.append(f'{round(np.average(VHL_ACC[i]), 4)}({round(calculate_variance(VHL_ACC[i]), 4)})')
                res.append(f'{round(np.average(VHL_PRE[i]), 4)}({round(calculate_variance(VHL_PRE[i]), 4)})')
                res.append(f'{round(np.average(VHL_PF[i]), 4)}({round(calculate_variance(VHL_PF[i]), 4)})')
                res.append(f'{round(np.average(VHL_PD[i]), 4)}({round(calculate_variance(VHL_PD[i]), 4)})')
                res.append(f'{round(np.average(VHL_F1[i]), 4)}({round(calculate_variance(VHL_F1[i]), 4)})')
                writer.writerow(res)
            res = []
            res.append(f'{round(np.average(JHL_ACC), 4)}({round(calculate_variance(JHL_ACC), 4)})')
            res.append(f'{round(np.average(JHL_PRE), 4)}({round(calculate_variance(JHL_PRE), 4)})')
            res.append(f'{round(np.average(JHL_PF), 4)}({round(calculate_variance(JHL_PF), 4)})')
            res.append(f'{round(np.average(JHL_PD), 4)}({round(calculate_variance(JHL_PD), 4)})')
            res.append(f'{round(np.average(JHL_F1), 4)}({round(calculate_variance(JHL_F1), 4)})')
            writer.writerow(res)
            res = []
            res.append(f'{round(np.average(J_VHL_ACC), 4)}({round(calculate_variance(J_VHL_ACC), 4)})')
            res.append(f'{round(np.average(J_VHL_PRE), 4)}({round(calculate_variance(J_VHL_PRE), 4)})')
            res.append(f'{round(np.average(J_VHL_PF), 4)}({round(calculate_variance(J_VHL_PF), 4)})')
            res.append(f'{round(np.average(J_VHL_PD), 4)}({round(calculate_variance(J_VHL_PD), 4)})')
            res.append(f'{round(np.average(J_VHL_F1), 4)}({round(calculate_variance(J_VHL_F1), 4)})')
            writer.writerow(res)

# if __name__ == '__main__':
#     for file in ['BGL.log'] :
#         for lamda in [1e-1, 0.5, 1, 5, 10]:
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
#                 pred_jvhl, vhl = joint_dev(embeddings, label[:len_t], label[len_tAd:], label[len_t : len_tAd], dev_label, id_dev, u, len_tAd, lamda=lamda, mu=1, gama=1)
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
#             result = 'record/_idea_lamda_mini.csv'
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
#     # 1e-1
#     # -----MVHL - ----
#     # TP = 1459204, TN = 1546742, FP = 56, FN = 0
#     # Acc = 99.9981, PRE = 99.9962, Recall = 100.0, F1 = 99.9981
#     # -----MVHL - ----
#     # TP = 1459204, TN = 1546797, FP = 1, FN = 0
#     # Acc = 100.0, PRE = 99.9999, Recall = 100.0, F1 = 99.9999
#     for file in ['Liberty.log'] :
#         for mu in []:
#             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
#             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
#             # HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
#             # VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], []], [[], []], [[], []], [[], []], [[], []]
#             HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
#             VHL_ACC, VHL_PRE, VHL_PF, VHL_PD, VHL_F1 = [[], [], []], [[], [], []], [[], [], []], [[], [], []], [[], [], []]
#             JHL_ACC, JHL_PRE, JHL_PF, JHL_PD, JHL_F1 = [], [], [], [], []
#             J_VHL_ACC, J_VHL_PRE, J_VHL_PF, J_VHL_PD, J_VHL_F1 = [99.9981, 100], [99.9962, 99.9999], [0.0036, 0.0], [100, 100], [99.9981, 99.9999]
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
#             for iter in range(8):
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
#                                            id_dev, u, len_tAd, lamda=1, mu=mu, gama=1)
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
#             result = 'record/_idea_mu_liberty.csv'
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
#
# if __name__ == '__main__':
#     for file in ['BGL.log'] :
#         for gama in [10]:
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
#                                            id_dev, u, len_tAd, lamda=1, mu=1, gama=gama)
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
#             result = 'record/_idea_gama_mini.csv'
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