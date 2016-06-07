# coding = GBK
'''
Created on 2016年6月1日

@author: coco1
'''
from numpy import* 
import operator
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis
from numpy.linalg.linalg import solve

'''
类定义
'''
class point:
    x = []
    y = []
        
'''
数据初始化
'''
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
def autoNorm(dataMat):
    minVal = dataMat.min(0)
    maxVal = dataMat.max(0)
    ranges = maxVal - minVal
    normDataSet = zeros(shape(dataMat))
    m = dataMat.shape[0]
    normDataSet = dataMat - tile(minVal , (m,1))
    normDataSet = normDataSet / (tile(ranges , (m,1)))
    print(normDataSet)
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
             
    

'''
点状图展示
    axes1.legend((type1), (u'19'), loc=2)
    axes2.legend((type2), (u'44'), loc=2)
    axes3.legend((type3), (u'51'), loc=2)
    axes4.legend((type4), (u'64'), loc=2)
    axes5.legend((type5), (u'115'), loc=2)
    axes6.legend((type6), (u'169'), loc=2)
    axes7.legend((type7), (u'194'), loc=2)
    axes8.legend((type8), (u'258'), loc=2)
    axes9.legend((type9), (u'260'), loc=2)
    axes10.legend((type10), (u'500'), loc=2)
    axes11.legend((type11), (u'601'), loc=2)
        axes1 = plt.subplot(111)
    axes2 = plt.subplot(111)
    axes3 = plt.subplot(111)
    axes4 = plt.subplot(111)
    axes5 = plt.subplot(111)
    axes6 = plt.subplot(111)
    axes7 = plt.subplot(111)
    axes8 = plt.subplot(111)
    axes9 = plt.subplot(111)
    axes10 = plt.subplot(111)
    axes11 = plt.subplot(111)
这段代码太丑了
'''

def plotData(group,label,labelbase):
    plt.figure(figsize=(16, 9), dpi=180)
    axes = plt.subplot(111)

    
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
    type3_x = []
    type3_y = []
    type4_x = []
    type4_y = []
    type5_x = []
    type5_y = []
    type6_x = []
    type6_y = []
    type7_x = []
    type7_y = []
    type8_x = []
    type8_y = []
    type9_x = []
    type9_y = []
    type10_x = []
    type10_y = []
    type11_x = []
    type11_y = []

    for i in range(len(group)):
        
        if label[i] == labelbase[0]: #1 type
            type1_x.append(group[i][0])
            type1_y.append(group[i][1])
        if label[i] == labelbase[1]: #2 type
            type2_x.append(group[i][0])
            type2_y.append(group[i][1])
        if label[i] == labelbase[2]: #3 type
            type3_x.append(group[i][0])
            type3_y.append(group[i][1])
        if label[i] == labelbase[3]: #4 type
            type4_x.append(group[i][0])
            type4_y.append(group[i][1])
        if label[i] == labelbase[4]: #5 type
            type5_x.append(group[i][0])
            type5_y.append(group[i][1])
        if label[i] == labelbase[5]: #6 type
            type6_x.append(group[i][0])
            type6_y.append(group[i][1])
        if label[i] == labelbase[6]: #7 type
            type7_x.append(group[i][0])
            type7_y.append(group[i][1])
        if label[i] == labelbase[7]: #8 type
            type8_x.append(group[i][0])
            type8_y.append(group[i][1])
        if label[i] == labelbase[8]: #9 type
            type9_x.append(group[i][0])
            type9_y.append(group[i][1])
        if label[i] == labelbase[9]: #10 type
            type10_x.append(group[i][0])
            type10_y.append(group[i][1])
        if label[i] == labelbase[10]: #11 type
            type11_x.append(group[i][0])
            type11_y.append(group[i][1])

               
    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=20, c='green')
    type3 = axes.scatter(type3_x, type3_y, s=20, c='black')
    type4 = axes.scatter(type4_x, type4_y, s=20, c='orange')
    type5 = axes.scatter(type5_x, type5_y, s=20, c='pink')
    type6 = axes.scatter(type6_x, type6_y, s=20, c='blue')
    type7 = axes.scatter(type7_x, type7_y, s=20, c='yellow')
    type8 = axes.scatter(type8_x, type8_y, s=20, c='brown')
    type9 = axes.scatter(type9_x, type9_y, s=20, c='grey')
    type10 = axes.scatter(type10_x, type10_y, s=20, c='cyan')
    type11 = axes.scatter(type11_x, type11_y, s=20, c='azure')
    
    plt.xlabel(u'lat')
    plt.ylabel(u'long')
    axes.legend((type1, type2 ,type3 ,type4 ,type5 ,type6 ,type7,type8 ,type9 ,type10 , type11), (u'19', u'44',u'51',u'64' ,u'115' ,u'169' ,u'194' ,u'258' ,u'260' , u'500' , u'601'), loc=2)

    
    plt.show()

'''
K-NN
'''
def classify(inX,dataSet,labels,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #tile:numpy中的函数。tile将原来的一个数组，扩充成了4个一样的数组。diffMat得到了目标与训练数值之间的差值。
    sqDiffMat   =diffMat**2#各个元素分别平方
    sqDistances =sqDiffMat.sum(axis=1)#对应列相加，即得到了每一个距离的平方
    distances   =sqDistances**0.5#开方，得到距离。
    sortedDistIndicies=distances.argsort()#升序排列
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    return sortedClassCount[0][0]
'''
算法评定
'''
def currentRate(labels , predictLabel):
    num = len(labels)
    currentNum = 0
    for i in range(num):
        if(labels[i] == predictLabel[i]):
            currentNum = currentNum + 1
        
    return currentNum / num
def countLabels(labels ,cata):
    type_1,type_2,type_3,type_4,type_5,type_6,type_7,type_8,type_9,type_10,type_11 = 0
    for i in range(len(labels)):
        if( labels[i] == cata[0]):
            type_1 = type_1 +1 
        if( labels[i] == cata[1]):
            type_2 = type_2 +1 
        if( labels[i] == cata[2]):
            type_3 = type_3 +1 
        if( labels[i] == cata[3]):
            type_4 = type_4 +1  
        if( labels[i] == cata[4]):
            type_5 = type_5 +1 
        if( labels[i] == cata[5]):
            type_6 = type_6 +1 
        if( labels[i] == cata[6]):
            type_7 = type_7 +1 
        if( labels[i] == cata[7]):
            type_8 = type_8 +1 
        if( labels[i] == cata[8]):
            type_9 = type_9 +1 
        if( labels[i] == cata[9]):
            type_10 = type_10 +1 
        if( labels[i] == cata[10]):
            type_11 = type_11 +1 
    return  type_1,type_2,type_3,type_4,type_5,type_6,type_7,type_8,type_9,type_10,type_11

def countTF(labels,predictLabels,cata,num):
    FP , TP , FN ,TN = 0
    for i in range(labels):
        if(labels[i] == predictLabels[i] & labels[i] == cata):
            TP = TP + 1
        if(labels[i] != predictLabels[i] & labels[i] == cata):
            FP = FP + 1
        if(labels[i] != predictLabels[i] & labels[i] != cata):
            FN = FN +1             
        if(labels[i] == predictLabels[i] & labels[i] != cata):
            TN = TN + 1        
    precision = TP /(TP + FP)
    recall = TP / (TP + FN)
    specifity = TN / (TN + FP)
    accuracy = (TN + TP) / (num)
    return precision , recall , specifity , accuracy
                
def fMeasure(r,a):
    f = (2 * r * a)/(r + a)   
    return f             

def f1Score(p,r):
    f = (2* p * r )/(p+r)
    return f       
'''
读取文件转换成矩阵
'''    
def txt2data(filename):
    fr = open(filename)
    arrayLines = fr.readlines()
    numOfLines = len(arrayLines)
    classLabelVector = []
    returnMat = zeros((numOfLines,2))
    
    index = 0
    for line in arrayLines:
        line = line.strip()
        listFromLine = line.split(' ')
        returnMat[index,0] = listFromLine[0]
        returnMat[index,1] = listFromLine[1]
        classLabelVector.append(int(listFromLine[-1])) #备用存储方案，用于绘图
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

'''
主程序区域
'''
'''
group,label = createDataSet()
print(len(group)) #len() and *.shape[0] , shape[0] is a easy way to measure matrix
plotData(group, label)
print(group.sum(axis=1))
'''
import samplingArchive 
#2015-11(9:00-24:00) data input
returnmat , classLabelVector = txt2data('D:\\FILE\\PythonWorkspace\\machineLearning\\data\\buffer.txt')

labels = txt2cata('D:\\FILE\\PythonWorkspace\\machineLearning\\data\\category.txt')



#数据集初始化
trS , trSL , tS , tSL = builtSet(returnmat, classLabelVector)

pTSL = []
for i in range(len(tS)):
    pTSL.append(classify(tS[i] ,trS , trSL , 11))
    cR = currentRate(pTSL , tSL)
print(cR)
#cT = countLabels(tSL, labels)
#pCT = countLabels(pTSL, labels) 
#plotData(returnmat,classLabelVector , labels)
#plotData(tS,pTSL , labels)
