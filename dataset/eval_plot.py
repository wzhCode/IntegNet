#-*- coding:UTF-8 -*-
import csv
import numpy as np
import matplotlib.pyplot as plt
from scipy import stats
import seaborn as sns
import pandas as pd

## plot the evaluation results.

def overall_evaluation():
    # fig.16: comparative evaluation of IntegNet with baseline methods.
    method = ['MA', 'EMWA', 'LSTM', 'GAT', 'IntegNet_h', 'IntegNet']
    overcost = [0.216, 0.173, 0.126, 0.125, 0.115, 0.073]
    undercost = [0.00205, 0.00166, 0.00162, 0.00208, 0.00182, 0.00260]

    # fig.17: performance of IntegNet for DC pairs with high traffic dynamics.
    # method = ['MA', 'EMWA', 'LSTM', 'GAT', 'IntegNet_h', 'IntegNet']
    # overcost = [0.262, 0.225, 0.150, 0.178, 0.139, 0.0795]
    # undercost = [0.00591, 0.00488, 0.00309, 0.00373, 0.00297, 0.00247]

    # fig.20: performance trade-off of IntegNet with different alpha in loss function.
    # method = [5,10,20,50,100]
    # overcost = [0.0388, 0.046, 0.0570, 0.0733, 0.0815]
    # undercost = [0.00748, 0.00592, 0.00403, 0.00260, 0.00217]

    # fig.21: benefits of GCN on the performance of IntegNet.
    # method = ['TCN_h', 'IntegNet_h', 'TCN', 'IntegNet']
    # overcost = [0.121, 0.115, 0.0719, 0.0733]
    # undercost = [0.00180, 0.00182, 0.00305, 0.00260]

    length = len(method)
    sns.set(context='paper', style='ticks')
    plt.rcParams['pdf.fonttype'] = 42
    fig = plt.figure(1, figsize=(5, 2.5), dpi=300)
    ax1 = fig.add_subplot(111)
    ax1.bar(range(length), overcost, color='b', label='Overcost')
    ax1.legend(loc=2, fontsize=8)
    ax1.set_ylabel('Overcost', fontsize=12)
    ax1.set_ylim([0.00, 0.3])
    # plt.xlabel(r'$\alpha$ in loss function', fontsize=12)

    ax2 = ax1.twinx()
    ax2.plot(range(length), undercost, color='r', marker='o', label='Undercost')
    ax2.set_xticks(range(length))
    ax2.set_xticklabels(method, fontsize=10)
    ax2.legend(loc=1, fontsize=8)
    ax2.set_ylabel('Undercost', fontsize=12)
    ax2.set_ylim([0, 0.01])

    plt.show()


def individual_evaluation():
    # fig.19: distribution of prediction performance over individual DC pairs.
    method = ['MA', 'EWMA', 'LSTM', 'GAT', 'IntegNet_h', 'IntegNet']
    length = len(method)
    overcost = [[] for i in range(length)]
    undercost = [[] for i in range(length)]
    fp = open('dcpair_predict_overcost_all_methods.csv', 'r')
    r = csv.reader(fp)
    index = 0
    for row in r:
        for item in row[1:]:
            overcost[index].append(float(item))
        index += 1
    fp.close()

    fp = open('dcpair_predict_undercost_all_methods.csv', 'r')
    r = csv.reader(fp)
    index = 0
    for row in r:
        for item in row[1:]:
            undercost[index].append(float(item))
        index += 1
    fp.close()

    sns.set(context='paper', style='whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(1, figsize=(5, 2.5), dpi=300)
    plt.boxplot(overcost, showfliers=False)
    plt.xticks(range(1, length + 1), method, rotation=0, size=10)
    plt.yticks(size=10)
    plt.ylabel('Overcost', size=12)
    plt.show()

    sns.set(context='paper', style='whitegrid')
    plt.rcParams['pdf.fonttype'] = 42
    plt.figure(2, figsize=(5, 2.5), dpi=300)
    plt.boxplot(undercost, showfliers=False)
    plt.xticks(range(1, length + 1), method, rotation=0, size=10)
    plt.yticks(size=10)
    plt.ylabel('Undercost', size=12)
    plt.show()



if __name__ == '__main__':
    print('start')
    overall_evaluation()
    # individual_evaluation()
    print('end')
