# -*- coding: utf-8 -*-
'''
Created on 2016年7月9日

@author: Shaow
'''
'''
读取文件转换成矩阵
'''    
from numpy import *
from poi import *
import os
import sys
import time
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
def txt2dataNum(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numOfLines = len(arrayLines)
    classLabelVector = []
    returnMat = zeros((numOfLines,2),double)
    num  = []
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(' ')
        returnMat[index,0] = listFromLine[0]
        returnMat[index,1] = listFromLine[1]
        classLabelVector.append(listFromLine[2]) #备用存储方案，用于绘图 numpy中的类无法做数组的游标
        num.append(listFromLine[3])
        index += 1
    return returnMat , classLabelVector ,num
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
def txt2list(filename):
    list = LinkList()
    fr = open(filename)
    arrayLines = fr.readlines()
    numOfLines = len(arrayLines)
    classLabelVector = []
    returnMat = zeros((numOfLines,2),double)

    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(' ')
        p = Poi(listFromLine[0],listFromLine[1],listFromLine[2],listFromLine[3])
        list.append(p)
        index += 1
    return LinkList
  
def save2txt(filepath,mat , labels):
    print(mat.shape[0] ,mat.shape[1])
    print(labels.shape[0] , labels.shape[1])
    savetxt(filepath + 'matSave',mat)
    savetxt(filepath + 'labelsSave',labels)
# start = time.clock()
# path = os.path.abspath(os.path.dirname(sys.argv[0]))
# list = LinkList()
# list = txt2list(path + '\\..\\data\\poi-test.txt')
# end = time.clock()
# print(end -start)