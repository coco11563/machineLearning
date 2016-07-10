# -*- coding: utf-8 -*-
'''
Created on 2016年7月9日

@author: Shaow
'''
'''
算法评定
'''
import numpy
def currentRate(labels , predictLabel):
    num = len(labels)
    pTStatus = []
    currentNum = 0
    for i in range(num):
        if(labels[i] == predictLabel[i]):
            currentNum = currentNum + 1
            pTStatus.append(1)
        else:
            pTStatus.append(0)
    
        
    return currentNum / num,pTStatus
def countLabels(labels ,cata):
    num = [0,0,0,0,0,0,0,0,0,0,0]
    for i in range(len(labels)):
        if( int(labels[i]) == int(cata[0])):
            num[0] = num[0] +1 
        if( int(labels[i]) == int(cata[1])):
            num[1] = num[1] +1 
        if( int(labels[i]) == int(cata[2])):
            num[2] = num[2] +1 
        if( int(labels[i]) == int(cata[3])):
            num[3] = num[3] +1  
        if( int(labels[i]) == int(cata[4])):
            num[4] = num[4] +1 
        if( int(labels[i]) == int(cata[5])):
            num[5] = num[5] +1 
        if( int(labels[i]) == int(cata[6])):
            num[6] = num[6] +1 
        if( int(labels[i]) == int(cata[7])):
            num[7] = num[7] +1 
        if( int(labels[i]) == int(cata[8])):
            num[8] = num[8] +1 
        if( int(labels[i]) == int(cata[9])):
            num[9] = num[9] +1  
        if( int(labels[i]) == int(cata[10])):
            num[10] = num[10] +1  
    return num

def countTF(labels,predictLabels,cata,pnum,num):
    FP = 0 #false positive
    TP = 0 #true positive
    FN = 0 #false negative
    TN = 0 #true negative
    for i in range(len(labels)):
        if(int(labels[i]) == int(predictLabels[i]) and int(labels[i]) == int(cata)):
            TP = TP + 1
        if(int(labels[i]) != int(predictLabels[i]) and int(labels[i]) == int(cata)):
            FN = FN + 1
        if(int(labels[i]) == int(predictLabels[i]) and int(labels[i] != cata)):
            FP = FP +1             
        if(int(labels[i]) != int(predictLabels[i]) and int(labels[i] != cata)):
            TN = TN + 1      
      
    precision = TP /(TP + FP)            #精确度：判定正例中真正正例的比重
    recall = TP / (TP + FN)              #召回率：判定正确的正例占总的正例的比重
    specifity = TN / (TN + FP)           #转移度：反映对0量的判定能力
    accuracy = (TN + TP) / (len(labels)) #准确度：反映分类器对整个样本的判断能力
    print('类别:' ,cata,'数量：',num,'precision(精确度：判定正例中真正正例的比重):',precision , 'recall(召回率：判定正确的正例占总的正例的比重):' , recall , 'specifity(转移度：反映对0量的判定能力):' , specifity , 'accuracy:(准确度：反映分类器对整个样本的判断能力)' , accuracy)
    return precision , recall , specifity , accuracy
                
def fMeasure(r,a):
    f = (2 * r * a)/(r + a) 
    print('this type\'s f-Measure is' , f)  
    return f             

def f1Score(p,r):
    f = (2* p * r )/(p+r)
    print('this type\'s f-1Score is' , f)
    return f 