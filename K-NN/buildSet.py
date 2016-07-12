# -*- coding: utf-8 -*-
'''
Created on 2016年7月9日

@author: Shaow
'''
'''
数据初始化
'''
from numpy import *
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    label = ['A','A','B','B']
    return group,label
'''
数据分集
输入原始数据集和其对应的标示
返回训练集、建立集、测试集
以及其相对应的标示
'''
def autoNorm(dataMat): #任意矩阵归一化
    minVal = dataMat.min(0)
    maxVal = dataMat.max(0)
    ranges = maxVal - minVal
    normDataSet = zeros(shape(dataMat))
    m = dataMat.shape[0]
    normDataSet = dataMat - tile(minVal , (m,1))
    normDataSet = normDataSet / (tile(ranges , (m,1)))
#    print(normDataSet)
    return normDataSet 
    
def unrepetitionRandomSampling(dataMat,number , labels):    #用以随机取样
    sample = zeros((number ,2))
    sampleLabels = []
    other = dataMat
    otherLabels = labels
    for i in range(number):
        randomnum =random.randint(0,len(other))
        sample[i,0] = other[randomnum,0]
        sample[i,1] = other[randomnum,1]
        sampleLabels.append(labels[randomnum])
        otherLabels = delete(otherLabels , randomnum )
        other = delete(other , randomnum , 0)
    return autoNorm(sample) ,sampleLabels, autoNorm(other),otherLabels
#无建立集的数据集生成
def builtSet(group,labels):
    num = len(group)
  
    trainSet = zeros((int(num * 0.7),2)) #train set 70%
    trainSetLabel = []

    testSet = zeros((num - int(num * 0.2),2)) #test set 30%
    testSetLabel = []
    trainSet,trainSetLabel, testSet,testSetLabel = unrepetitionRandomSampling(group ,int(num * 0.7) ,labels )
    return trainSet,trainSetLabel, testSet,testSetLabel