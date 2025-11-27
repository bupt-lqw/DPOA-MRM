from sklearn.svm import SVC
from sklearn.metrics import recall_score, f1_score, confusion_matrix, roc_auc_score, matthews_corrcoef
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from flow import train, getFlowPredictRes
from getOverlap import getTestSamplesInsOverlapDegree
import numpy as np
import pandas as pd
import math


def getSplitTestData(overlapData, noOverlapData, test_data):
    overlapX, overlapY = overlapData[:, :-1], overlapData[:, -1]
    noOverlapX, noOverlapY = noOverlapData[:, :-1], noOverlapData[:, -1]
    
    k = 5
    overlapKnn = KNeighborsClassifier(n_neighbors=k).fit(overlapX, overlapY)
    noOverlapKnn = KNeighborsClassifier(n_neighbors=k).fit(noOverlapX, noOverlapY)
    overlapDis, _ = overlapKnn.kneighbors(test_data[:, :-1])
    noOverlapDis, _ = noOverlapKnn.kneighbors(test_data[:, :-1])
    overlapDis, noOverlapDis = np.mean(overlapDis, axis=1), np.mean(noOverlapDis, axis=1)
    length = test_data.shape[0]
    overlapIndex, noOverlapIndex = [], []
    for i in range(length):
        if overlapDis[i] < noOverlapDis[i]:
            overlapIndex.append(i)
        else:
            noOverlapIndex.append(i)
    return test_data[overlapIndex, :], test_data[noOverlapIndex, :]

def getKnnAvgDis(data, oneSample, k):
    try:
        knn = KNeighborsClassifier(n_neighbors=k).fit(data[:, :-1], data[:, -1])
        diss, _ = knn.kneighbors(oneSample[:, :-1])
        return np.mean(diss, axis=1).reshape(1, 1)[0, 0]
    except:
        return 0

def isOverlapTestData(overlapData, noOverlapData, oneTestSample, k):
    return True if getKnnAvgDis(overlapData, oneTestSample, k) < getKnnAvgDis(noOverlapData, oneTestSample, k) else False

def getProbFromFea(feaOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapDatas, oneTestSample, feaNoOverlapClf, args):
    prob = 0
    if isOverlapTestData(feaOverlapDatas, feaNoOverlapDatas, oneTestSample, args.isOverlapTestSampleK) and transferredFeaOverlapDatas is not None:
        prob = getFlowPredictRes(transferredFeaOverlapDatas, oneTestSample, model, args)
    else:
        prob = feaNoOverlapClf.predict_proba(oneTestSample[:, :-1])[0, -1]
    return prob


def getProbFromFea_wo_flow(feaOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapDatas, oneTestSample, feaNoOverlapClf, args):
    # prob = 0
    # if isOverlapTestData(feaOverlapDatas, feaNoOverlapDatas, oneTestSample, args.isOverlapTestSampleK):
    #     prob = getFlowPredictRes(transferredFeaOverlapDatas, oneTestSample, model, args)
    # else:
    #     prob = feaNoOverlapClf.predict_proba(oneTestSample[:, :-1])[0, -1]
    return feaNoOverlapClf.predict_proba(oneTestSample[:, :-1])[0, -1]


def getOverlapAndNoOverlapDatas(trainData, overlapIndexs):  # 根据交叠样本的index获取交叠样本和非交叠样本
    train_data_num = trainData.shape[0]
    allIndexs = [i for i in range(train_data_num)]
    noOverlapIndexs = list(set(allIndexs) - set(overlapIndexs))
    overlapData = trainData[overlapIndexs, :]
    noOverlapData = trainData[noOverlapIndexs, :]
    return overlapIndexs, noOverlapIndexs, overlapData, noOverlapData

def splitTrainData(train_data, overlapDegrees, threshold):
    overlapIndexs = []
    for i in range(len(overlapDegrees)):
        if overlapDegrees[i] > threshold:
            overlapIndexs.append(i)
    overlapIndexs, noOverlapIndexs, overlapData, noOverlapData = getOverlapAndNoOverlapDatas(train_data, overlapIndexs)
    return overlapIndexs, noOverlapIndexs, overlapData, noOverlapData

def useFea(feaOverlapDatas):
    num1 = feaOverlapDatas[feaOverlapDatas[:, -1] == 1].shape[0]
    num0 = feaOverlapDatas[feaOverlapDatas[:, -1] == 0].shape[0]
    if num0 < 4 or num1 < 4:
        return False
    return True

def getClf(classifier_name):
    if classifier_name == 'rf':
        return RandomForestClassifier()
    if classifier_name == 'lr':
        return LogisticRegression()
    if classifier_name == 'svm':
        return SVC(probability=True)
    if classifier_name == 'gbdt':
        return GradientBoostingClassifier()
    return None

def getModel(train_data, featureOverlapDegrees, instanceOverlapDegrees, args):
    _, _, feaOverlapDatas, feaNoOverlapDatas = splitTrainData(train_data, featureOverlapDegrees, args.feaOverlapThreshold)
    feaNoOverlapClf = getClf(args.classifier_name).fit(feaNoOverlapDatas[:, :-1], feaNoOverlapDatas[:, -1])
    transferredFeaOverlapDatas, model = None, None
    if useFea(feaOverlapDatas):
        transferredFeaOverlapDatas, model = train(feaOverlapDatas, args)
    
    insDatas = np.concatenate((train_data[:, :-1], np.array(instanceOverlapDegrees).reshape(-1, 1), train_data[:, -1].reshape(-1, 1)), axis=1)
    insClf = getClf(args.classifier_name).fit(insDatas[:, :-1], insDatas[:, -1])
    return feaOverlapDatas, feaNoOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapClf, insClf

def getModel_only_ins(train_data, featureOverlapDegrees, instanceOverlapDegrees, args):
    _, _, feaOverlapDatas, feaNoOverlapDatas = splitTrainData(train_data, featureOverlapDegrees, args.feaOverlapThreshold)
    feaNoOverlapClf = getClf(args.classifier_name).fit(feaNoOverlapDatas[:, :-1], feaNoOverlapDatas[:, -1])
    transferredFeaOverlapDatas, model = None, None
    # if useFea(feaOverlapDatas):
    #     transferredFeaOverlapDatas, model = train(feaOverlapDatas, args)
    
    insDatas = np.concatenate((train_data[:, :-1], np.array(instanceOverlapDegrees).reshape(-1, 1), train_data[:, -1].reshape(-1, 1)), axis=1)
    insClf = getClf(args.classifier_name).fit(insDatas[:, :-1], insDatas[:, -1])
    return feaOverlapDatas, feaNoOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapClf, insClf



def getProbs(train_data, test_data, feaOverlapDatas, feaNoOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapClf, insClf, args, index):
    testDatasOverlapDegrees = getTestSamplesInsOverlapDegree(train_data, test_data, args.ks[index])
    testDatas = np.concatenate((test_data[:, :-1], np.array(testDatasOverlapDegrees).reshape(-1, 1), test_data[:, -1].reshape(-1, 1)), axis=1)
    insProbs = insClf.predict_proba(testDatas[:, :-1])[:, -1]
    # model = None  # 只使用实例交叠
    if model != None:
        allPorbs = []
        for i in range(test_data.shape[0]):
            testSample = test_data[i, :].reshape(1, -1)
            feaProb = getProbFromFea(feaOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapDatas, testSample, feaNoOverlapClf, args)
            insProb = insProbs[i]
            prob = feaProb * args.w_feas[index] + insProb * args.w_inss[index]
            allPorbs.append(prob)
        
        allPorbs = np.array(allPorbs).reshape(-1, 1)
        # feaProbs = np.array(feaProbs).reshape(-1, 1)
        return allPorbs
    return insProbs.reshape(-1, 1)


def getProbs_wo_flow(train_data, test_data, feaOverlapDatas, feaNoOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapClf, insClf, args, index):
    testDatasOverlapDegrees = getTestSamplesInsOverlapDegree(train_data, test_data, args.ks[index])
    testDatas = np.concatenate((test_data[:, :-1], np.array(testDatasOverlapDegrees).reshape(-1, 1), test_data[:, -1].reshape(-1, 1)), axis=1)
    insProbs = insClf.predict_proba(testDatas[:, :-1])[:, -1]
    # model = None  # 只使用实例交叠
    if model != None:
        allPorbs = []
        for i in range(test_data.shape[0]):
            testSample = test_data[i, :].reshape(1, -1)
            feaProb = getProbFromFea_wo_flow(feaOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapDatas, testSample, feaNoOverlapClf, args)
            insProb = insProbs[i]
            prob = feaProb * args.w_feas[index] + insProb * args.w_inss[index]
            allPorbs.append(prob)
        
        allPorbs = np.array(allPorbs).reshape(-1, 1)
        # feaProbs = np.array(feaProbs).reshape(-1, 1)
        return allPorbs
    return insProbs.reshape(-1, 1)



def getProbs_only_fea(train_data, test_data, feaOverlapDatas, feaNoOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapClf, insClf, args, index):
    # testDatasOverlapDegrees = getTestSamplesInsOverlapDegree(train_data, test_data, args.ks[index])
    # testDatas = np.concatenate((test_data[:, :-1], np.array(testDatasOverlapDegrees).reshape(-1, 1), test_data[:, -1].reshape(-1, 1)), axis=1)
    # insProbs = insClf.predict_proba(testDatas[:, :-1])[:, -1]
    # model = None  # 只使用实例交叠
    # if model != None:
    allPorbs = []
    for i in range(test_data.shape[0]):
        testSample = test_data[i, :].reshape(1, -1)
        feaProb = getProbFromFea(feaOverlapDatas, transferredFeaOverlapDatas, model, feaNoOverlapDatas, testSample, feaNoOverlapClf, args)
        # insProb = insProbs[i]
        prob = feaProb * args.w_feas[index]
        allPorbs.append(prob)
    
    allPorbs = np.array(allPorbs).reshape(-1, 1)
    # feaProbs = np.array(feaProbs).reshape(-1, 1)
    return allPorbs
    # return insProbs.reshape(-1, 1)

def getBestThreshold(prob, testY):
    bestThreshold = 0
    maxV = 0
    for t in range(0, 100):
        threshold = 1.0 * t / 100
        predY = np.where(prob >= threshold, 1, 0)
        f1 = f1_score(testY, predY, average=None)[1]
        recall = recall_score(testY, predY, average=None)
        gmean = math.sqrt(recall[0] * recall[1])
        v = f1 + gmean
        if v > maxV:
            maxV = v
            bestThreshold = threshold
    return bestThreshold

def getP(testProb, testY, valProb, valY):
    threshold = getBestThreshold(valProb, valY)
    predY = np.where(testProb > threshold, 1, 0)
    f1 = f1_score(testY, predY, average=None)[1]
    recall = recall_score(testY, predY, average=None)
    gmean = math.sqrt(recall[0] * recall[1])
    tn, fp, fn, tp = confusion_matrix(testY, predY).ravel()
    return f1, gmean, tn, fp, fn, tp

def calP(allTestProbs, allTestY, allValProbs, allValY, args):
    performance_df = pd.DataFrame(columns=['classifer', 'f1', 'gmean', 'TN', 'FP', 'FN', 'TP'])  # 初始化性能表现df
    for i in range(len(allTestProbs)):
        f1, gmean, auc, mcc, tn, fp, fn, tp = getP(allTestProbs[i], allTestY[i], allValProbs[i], allValY[i])
        performance_df.loc[len(performance_df)] = [args.classifier_name, f1, gmean, auc, mcc, tn, fp, fn, tp]
    return performance_df

def getSubPred(valY, testProbs, valProbs):
    threshold = getBestThreshold(valProbs, valY)
    predY = np.where(testProbs > threshold, 1, 0)
    valPredY = np.where(valProbs > threshold, 1, 0)
    return predY, valPredY

def getPFromPred(predY, valPredY, testY, testValY):
    print(predY.shape, valPredY.shape, testY.shape, testValY.shape)
    threshold = getBestThreshold(valPredY, testValY)
    predY = np.where(predY > threshold, 1, 0)
    f1 = f1_score(testY, predY, average=None)[1]
    recall = recall_score(testY, predY, average=None)
    gmean = math.sqrt(recall[0] * recall[1])
    tn, fp, fn, tp = confusion_matrix(testY, predY).ravel()
    
    return f1, gmean, tn, fp, fn, tp
