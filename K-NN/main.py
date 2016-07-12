'''
Created on 2016年7月9日

@author: Shaow
'''
from numpy import * 
import operator
from numpy.linalg.linalg import solve
from math import *
# from progressBar import *
import time
import sys        
import measure
import samplingArchive 
import os
from matplotlibPlot import *
from readFile import *
from knn import *
from measure import *
from poi import *
if __name__ == '__main__':
    path = os.path.abspath(os.path.dirname(sys.argv[0]))
    print('正在初始化参数...')
#2015-11(9:00-24:00) data input
    testmat , testclassLabelVector = txt2data(path + '\\..\\data\\buffer.txt')
#testmat = autoNorm(testmat) 使用norm后正确率22%
    returnmat , classLabelVector ,numofcheck = txt2dataNum(path + '\\..\\data\\poi-test.txt')
    #returnmat , classLabelVector  = txt2data(path + '\\..\\data\\8buffer.txt')
#returnmat = autoNorm(returnmat)
#save2txt('D:\\FILE\\PythonWorkspace\\machineLearning\\data\\', testmat, testclassLabelVector)
    labels = txt2cata(path + '\\..\\data\\category.txt')
    time.sleep(1)
    print('矩阵初始化完成！')
    pTSL = []
#     bar = ProgressBar(total = 100)
#     print('----------三秒后展示原坐标点图----------')
#     barProcess(1,3)
    #plotData(returnmat,classLabelVector , labels)
#     print('原坐标图展示完成！')
    num = len(testmat)
#     print('----------K-NN计算进展情况----------')
    
    for i in range(len(testmat)):
        if (i % int(len(testmat)/100)  == 0):
            print('done 1%')
#         if(i%(int(len(testmat)/100))==0): 
#             bar.log('We have arrived at: ' + str(i + 1))
#             bar.move()
        pTSL.append(classifyByPoi(testmat[i] ,returnmat , classLabelVector ,numofcheck, 5))
        #pTSL.append(classify(testmat[i] ,returnmat , classLabelVector , 11))
    
    cR,pStatus  = currentRate(pTSL , testclassLabelVector)
    print('\n')
    print('本次预测准确度为:',cR) 
    cT = countLabels(testclassLabelVector, labels)
    pCT = countLabels(pTSL, labels) 
    f1score = []
    fm  = []
    for j in range(len(labels)):
        
        p,r,s,a = countTF(testclassLabelVector,pTSL,labels[j] , pCT[j],cT[j])
        fm.append(fMeasure(r,a))
        f1score.append(f1Score(p,r))
#     print('----------三秒后展示预测坐标点图----------')
#     barProcess(1,3)
    plotData(testmat , pTSL , labels)
    print('预测图展示完成！')
#     print('----------三秒后展示预测情况图----------')
#     barProcess(1,3)
    plotData_2(testmat , pStatus)
    print('情况图展示完成！')
