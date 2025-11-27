from dataPreprocess import load_dataset, get_train_test_list, save_data
from getOverlap import instanceOverlap, featureOverlap
from getPerformance import getProbs, calP, getModel, getSubPred, getPFromPred
import argparse
import numpy as np
import pandas as pd
import math
import os

def ags_parse():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_path', default='./datasets/')
    parser.add_argument('--save_data_path', default='./saved_data/')
    parser.add_argument('--model_path', default='./models/')
    parser.add_argument('--each_performance_path', default='./each_performance/')
    parser.add_argument('--combine_performance_path', default='performance.csv')
    parser.add_argument('--device', default='cuda:0')
    # 划交叠区参数
    parser.add_argument('--ks', default=[1,3,5,7,9])
    parser.add_argument('--w_feas', default=[0.9,0.7,0.5,0.3,0.1])
    parser.add_argument('--w_inss', default=[0.1,0.3,0.5,0.7,0.9])
    parser.add_argument('--feaOverlapThreshold', default=0.5)
    parser.add_argument('--insOverlapThreshold', default=0.5)
    parser.add_argument('--isOverlapTestSampleK', default=5)
    # 流模型参数
    parser.add_argument('--epoch', default=50, type=int)
    parser.add_argument('--batch_size', default=16, type=int)
    parser.add_argument('--n_blocks', default=4, type=int, help='depth of the flow model')
    parser.add_argument('--n_flows', default=1, type=int, help='depth of the glow model')
    parser.add_argument('--lr', default=1e-4, type=float, help='learning rate')
    parser.add_argument('--k', default=5, type=int, help='knns')
    # 分类器
    parser.add_argument('--classifier_name', default='rf')
    args = parser.parse_args()
    return args

def calFeaOverlapThreshold(train_data, args):
    num0 = train_data[train_data[:, -1] == 0].shape[0]
    num1 = train_data[train_data[:, -1] == 1].shape[0]
    ir = 1.0 * num0 / num1
    # args.feaOverlapThreshold = 0.5 * (1.0 / math.sqrt(ir))
    args.feaOverlapThreshold = 0.5 * (1.0 / (ir))

def train_model(data_name):
    args = ags_parse()
    print('*' * 30, data_name, ' start', '*' * 30)
    # 加载数据集并五折
    data = load_dataset(args.data_path, data_name)
    five_train_test_list = get_train_test_list(data)
    performance_df = pd.DataFrame(columns=['classifer', 'f1', 'gmean'])
    
    for i in range(5):
        predList = []
        valPredList = []
        F1List, GmeanList = [], []
        
        print('*' * 30, data_name, ' ' * 10, str(i+1), ' ' * 10, '*' * 30)        
        # 数据准备
        train_data, test_data, val_data = five_train_test_list[i][0], five_train_test_list[i][1], five_train_test_list[i][2]
        calFeaOverlapThreshold(train_data, args)
        save_data(args.save_data_path, i, data_name, train_data, test_data, val_data)  # 保存数据
        for index in range(len(args.ks)):
            # 分别获取样本的实例交叠程度和特征交叠程度
            instanceOverlapDegrees, featureOverlapDegrees = instanceOverlap(train_data, args, index), featureOverlap(train_data, args, index)
            # 高分辨率（k比较小）时，应该重点关注特征交叠；低分辨率（k比较大）时，应该重点关注实例交叠
            # 现在是顺序是k从小到大，即分辨率从高到低
            # 对特征交叠的样本使用流模型进行特征降维、对实例交叠的样本赋予更大的权重
            feaOverlapDatas, feaNoOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapClf, insClf = getModel(train_data, featureOverlapDegrees, instanceOverlapDegrees, args)
            testProbs = getProbs(train_data, test_data, feaOverlapDatas, feaNoOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapClf, insClf, args, index)
            valProbs = getProbs(train_data, val_data, feaOverlapDatas, feaNoOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapClf, insClf, args, index)
            predList.append(testProbs)
            valPredList.append(valProbs)
            f1, gmean, _, _, _, _ = getPFromPred(testProbs, valProbs, test_data[:, -1].reshape(-1, 1), val_data[:, -1].reshape(-1, 1))
            F1List.append(f1)
            GmeanList.append(gmean)
            
        predY = np.mean(np.array(predList).squeeze(), axis=0)
        valPredY = np.mean(np.array(valPredList).squeeze(), axis=0)
        f1, gmean, _, _, _, _ = getPFromPred(predY, valPredY, test_data[:, -1].reshape(-1, 1), val_data[:, -1].reshape(-1, 1))
        
