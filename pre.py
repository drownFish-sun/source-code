import csv

import numpy as np
import pandas as pd
# def read_in_chunks(file_object, chunk_size=10000):
#     while True:
#         try:
#             lines = [file_object.readline().strip() for _ in range(chunk_size)]
#             if not lines:
#                 break
#             yield lines
#         except Exception as e:
#             continue
# with open('dataset/liberty/liberty.log', 'a+') as writer:
#     illegal = 0
#     flag = 0
#     with open('dataset/liberty/liberty2', 'r') as f:
#         count = 0
#         for chunk in read_in_chunks(f, chunk_size=10000 if flag == 0 else illegal):
#             count+=1
#             if count < 10000:
#                 continue
#             for log in chunk:
#                 if len(log.split(' ')) > 8:
#                     writer.writelines(log.strip() + '\n')
#                 else:
#                     illegal += 1
#             print(count)
#             if count == 11000:
#                 flag = 1
#             if count > 11000 and flag:
#                 break

HL_ACC, HL_PRE, HL_PF, HL_PD, HL_F1 = [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []], [[], [], [], []]
count = 0
flag = -1
#Acc=99.9537, PRE=99.3783, Recall=100.0, F1=99.6882
with open('2.txt', 'r') as f:
    for line in f.readlines():
        if line.startswith('-----VHL-----'):
            count = 1
        if count == 1 and line.startswith('Acc='):
            flag += 1
            flag %= 4
            _ = line.strip().split(', ')
            HL_ACC[flag].append(eval(_[0].split('=')[1]))
            HL_PRE[flag].append(eval(_[1].split('=')[1]))
            HL_PD[flag].append(eval(_[2].split('=')[1]))
            HL_F1[flag].append(eval(_[3].split('=')[1]))
            if flag == 3:
                count = 0
    for i in range(4):
        print(np.round(np.average(HL_ACC[i]), 4), np.round(np.average(HL_PRE[i]), 4), np.round(np.average(HL_PD[i]), 4), np.round(np.average(HL_F1[i]), 4))