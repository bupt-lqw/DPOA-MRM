from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
import numpy as np
from sklearn.preprocessing import MinMaxScaler

def getF1(data):
    x, y = data[:, :-1], data[:, -1]
    maj = x[y == 0]
    min = x[y == 1]
    record = []
    for i in range(x.shape[1]):
        mu0 = np.mean(maj[:, i])
        mu1 = np.mean(min[:, i])
        var0 = np.var(maj[:, i])
        var1 = np.var(maj[:, i])
        
        fi = (mu0 - mu1) ** 2 / (var0 + var1)
        if var0 == 0 and var1 == 0:
            fi = (mu0 - mu1) ** 2 / 0.0001
        record.append(fi)
    # re = max(record)
    # re = 1 / (re + 1)
    return record

def getWeights(data):
    x, y = data[:, :-1], data[:, -1]
    rf = RandomForestClassifier(n_estimators=100, random_state=1)
    rf.fit(x, y)
    weights = []
    # 获取特征重要性
    importances = rf.feature_importances_
    for _, importance in enumerate(importances):
        weights.append(importance)
    return weights

def getTopFeas(importances):
    # 假设我们有一个总和为1的list
    # 将importances按照从大到小排序，并保留对应的索引
    sorted_importances = sorted(enumerate(importances), key=lambda x: x[1], reverse=True)

    # 初始化累计和和子序列的索引列表
    cumulative_sum = 0
    subsequence_indices = []
    sumV = sum(importances)
    # 遍历排序后的importances
    for index, importance in sorted_importances:
        # 如果加上当前的重要性后累计和不超过0.9*sum，则加入子序列
        if cumulative_sum + importance <= 0.9 * sumV:
            cumulative_sum += importance
            subsequence_indices.append(index)
        # 如果超过了0.9，则停止
        else:
            break
    return subsequence_indices

def featureOverlap(data, args, index):
    weights = getWeights(data)
    f1s = getF1(data)
    importances = [weights[i] * f1s[i] for i in range(len(weights))]
    indics = getTopFeas(importances)
    allOverlapDegrees = []
    for i in indics:
        subData = np.hstack((data[:, i].reshape(-1, 1), data[:, -1].reshape(-1, 1)))
        subOverlapDeg = instanceOverlap(subData, args, index)
        allOverlapDegrees.append(subOverlapDeg)
    allOverlapDegrees = np.array(allOverlapDegrees).T
    
    wei = np.array([importances[i] for i in indics]).reshape(-1, 1)
    sumW = np.sum(wei)
    wei = wei / sumW
    
    overlapDegrees = np.dot(allOverlapDegrees, wei)
    overlapDegrees.reshape(-1,)
    # maxInportanceIndex = weights.index(max(weights))
    # subData = np.hstack((data[:, maxInportanceIndex].reshape(-1, 1), data[:, -1].reshape(-1, 1)))
    # overlapDegrees = instanceOverlap(subData, args, index)
    
    return [overlapDegrees[i] for i in range(len(overlapDegrees))]

def instanceOverlap(data, args, index):
    x, y = data[:, :-1], data[:, -1]
    k = args.ks[index]
    knn = KNeighborsClassifier(n_neighbors=k+1).fit(x, y)
    overlapDegrees = []
    for i in range(x.shape[0]):
        _, indices = knn.kneighbors(x[i, :].reshape(1, -1))  # 获取近邻样本（包括本身）的距离和索引
        indices = indices[0][1:]
        kNeiAvgLabel = 1.0 * sum(y[indices]) / k
        overlapDegrees.append(abs(y[i] - kNeiAvgLabel)) # 平均近邻标签与实际标签差异越大，越有可能位于交叠区
    return overlapDegrees

def getTestSamplesInsOverlapDegree(data, test_data, k):
    x, y = data[:, :-1], data[:, -1]
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(x, y)
    overlapDegrees = []
    for i in range(test_data.shape[0]):
        _, indices = knn.kneighbors(test_data[i, :-1].reshape(1, -1))  # 获取近邻样本（包括本身）的距离和索引
        indices = indices[0]
        kNeiAvgLabel = 1.0 * sum(y[indices]) / k
        overlapDegrees.append(abs(y[i] - kNeiAvgLabel))  # 平均近邻标签与实际标签差异越大，越有可能位于交叠区
    return overlapDegrees



def multiresolutionFeatureOverlap(data, ks=[3,5,7,9,11]):
    feaOverlapRecord = []
    for k in ks:
        featureOverlapIndexs = featureOverlap(data, k)
        feaOverlapRecord.append(featureOverlapIndexs)
    feaOverlapRecord = np.array(feaOverlapRecord).T
    feaOverlapDegree = np.mean(feaOverlapRecord, axis=1)
    
    x, y = data[:, :-1], data[:, -1].reshape(-1, 1)
    length = data.shape[0]
    isOverlapIndexs = [0 for _ in range(length)]  # 0: 不是交叠区样本    1: 是交叠区样本
    featureOverlapIndexs = []
    for i in range(length):
        if feaOverlapDegree[i] > 0.5:
            isOverlapIndexs[i] = 1
            featureOverlapIndexs.append(i)

    return featureOverlapIndexs


def multiresolutionInstanceOverlap(data, ks=[3,5,7,9,11]):
    feaOverlapRecord = []
    for k in ks:
        featureOverlapIndexs = instanceOverlap(data, k)
        feaOverlapRecord.append(featureOverlapIndexs)
    feaOverlapRecord = np.array(feaOverlapRecord).T
    feaOverlapDegree = np.mean(feaOverlapRecord, axis=1)
    
    x, y = data[:, :-1], data[:, -1].reshape(-1, 1)
    length = data.shape[0]
    isOverlapIndexs = [0 for _ in range(length)]  # 0: 不是交叠区样本    1: 是交叠区样本
    featureOverlapIndexs = []
    for i in range(length):
        if feaOverlapDegree[i] > 0.5:
            isOverlapIndexs[i] = 1
            featureOverlapIndexs.append(i)

    return featureOverlapIndexs









