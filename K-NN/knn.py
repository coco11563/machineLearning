# coding = UTF-8
'''
Created on 2016年6月1日

@author: coco1
'''
from numpy import*
import operator
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis

'''
数据初始化
'''
def createDataSet():
    group = array([[1.0,1.1],[1.0,1.0],[0,0],[0,0.1]])
    lable = ['A','A','B','B']
    return group,lable

'''
点状图展示
'''

def plotData(group,lable):
    plt.figure(figsize=(8, 5), dpi=80)
    axes = plt.subplot(111)
    typeA_x = []
    typeA_y = []
    typeB_x = []
    typeB_y = []
    for i in range(len(group)):
        if lable[i] == 'A': #A type
            typeA_x.append(group[i][0])
            typeA_y.append(group[i][1])
        if lable[i] == 'B': #B type
            typeB_x.append(group[i][0])
            typeB_y.append(group[i][1])
    typeA = axes.scatter(typeA_x, typeA_y, s=20, c='red')
    typeB = axes.scatter(typeB_x, typeB_y, s=20, c='green')
    plt.xlabel(u'x')
    plt.ylabel(u'y')
    axes.legend((typeA, typeB), (u'A', u'B'), loc=2)
    plt.show()

'''
K-NN
'''
def classify(inX,dataset,labels,k):
    dataSetSize = dataSet.shape[0]
    diffmat = tile(inX,(dataSetSize,1)) - dataset #tile:numpy中的函数。tile将原来的一个数组，扩充成了4个一样的数组。diffMat得到了目标与训练数值之间的差值。
    sqDiffMat   =diffMat**2#各个元素分别平方
    sqDistances =sqDiffMat.sum(axis=1)#对应列相加，即得到了每一个距离的平方
    distances   =sqDistances**0.5#开方，得到距离。
    sortedDistIndicies=distances.argsort()#升序排列
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #排序
    sortedClassCount=sorted(classCount.iteritems(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
    
'''
主程序区域
'''
group,lable = createDataSet()
print(len(group)) #len() and *.shape[0] , shape[0] is a easy way to measure matrix
plotData(group, lable)
print(group.sum(axis=1))



        
    