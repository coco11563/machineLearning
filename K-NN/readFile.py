# -*- coding: utf-8 -*-
'''
Created on 2016年7月9日

@author: Shaow
'''
'''
读取文件转换成矩阵
'''    
from numpy import *
def txt2data(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numOfLines = len(arrayLines)
    classLabelVector = []
    returnMat = zeros((numOfLines,2),double)

    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(' ')
        returnMat[index,0] = listFromLine[0]
        returnMat[index,1] = listFromLine[1]
        classLabelVector.append(listFromLine[2]) #备用存储方案，用于绘图 numpy中的类无法做数组的游标
        index += 1
    return returnMat , classLabelVector
'''
读取类别
'''
def txt2cata(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numOfLines = len(arrayLines)
    labels = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        labels.append(int(line))
        index += 1
    return labels

def save2txt(filepath,mat , labels):
    print(mat.shape[0] ,mat.shape[1])
    print(labels.shape[0] , labels.shape[1])
    savetxt(filepath + 'matSave',mat)
    savetxt(filepath + 'labelsSave',labels)
    