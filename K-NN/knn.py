# coding = GBK
'''
Created on 2016年6月1日

@author: coco1
'''
from numpy import* 
import distanceCacu as cacu
import operator
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, spectral
from numpy.linalg.linalg import solve
from math import *
import time
import sys



'''
类定义
'''
class point:
    x = []
    y = []
class ProgressBar:
    def __init__(self, count = 0, total = 0, width = 50):
        self.count = count
        self.total = total
        self.width = width
    def move(self):
        self.count += 1
    def log(self, s):
        sys.stdout.write('进度条：')
        sys.stdout.write(' ' * (self.width + 9) + '\r')
        sys.stdout.flush()
        progress = int(self.width * self.count / self.total)
        sys.stdout.write('{0:3}/{1:3}: '.format(self.count, self.total))
        sys.stdout.write('O' * progress + 'o' * int(self.width - progress) + '\r')
        if progress == self.width:
            sys.stdout.write('\n')
        sys.stdout.flush()


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

                
def barProcess(times , num ):   
    bar = ProgressBar(total = num)
    for i in range(num):
        bar.move()
        bar.log(' ')
        time.sleep(times)
        
    
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
        
        if int(label[i]) == labelbase[0]: #1 type
            type1_x.append(group[i][0])
            type1_y.append(group[i][1])
        if int(label[i]) == labelbase[1]: #2 type
            type2_x.append(group[i][0])
            type2_y.append(group[i][1])
        if int(label[i]) == labelbase[2]: #3 type
            type3_x.append(group[i][0])
            type3_y.append(group[i][1])
        if int(label[i]) == labelbase[3]: #4 type
            type4_x.append(group[i][0])
            type4_y.append(group[i][1])
        if int(label[i]) == labelbase[4]: #5 type
            type5_x.append(group[i][0])
            type5_y.append(group[i][1])
        if int(label[i]) == labelbase[5]: #6 type
            type6_x.append(group[i][0])
            type6_y.append(group[i][1])
        if int(label[i]) == labelbase[6]: #7 type
            type7_x.append(group[i][0])
            type7_y.append(group[i][1])
        if int(label[i]) == labelbase[7]: #8 type
            type8_x.append(group[i][0])
            type8_y.append(group[i][1])
        if int(label[i]) == labelbase[8]: #9 type
            type9_x.append(group[i][0])
            type9_y.append(group[i][1])
        if int(label[i]) == labelbase[9]: #10 type
            type10_x.append(group[i][0])
            type10_y.append(group[i][1])
        if int(label[i]) == labelbase[10]: #11 type
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
def plotData_2(group,label):
    plt.figure(figsize=(16, 9), dpi=180)
    axes = plt.subplot(111) 
    type1_x = []
    type1_y = []
    type2_x = []
    type2_y = []
  
    for i in range(len(group)):
        
        if int(label[i]) == 0: #1 type
            type1_x.append(group[i][0])
            type1_y.append(group[i][1])
        if int(label[i]) == 1: #2 type
            type2_x.append(group[i][0])
            type2_y.append(group[i][1])
       
         
    type1 = axes.scatter(type1_x, type1_y, s=20, c='red')
    type2 = axes.scatter(type2_x, type2_y, s=20, c='cyan')
    
    
    plt.xlabel(u'lat')
    plt.ylabel(u'long')
    axes.legend((type1, type2), (u'wrong', u'right'), loc=2)

   
    plt.show()

'''
经纬度计算距离
'''
def calcDistance(Lat_A, Lng_A, Lat_B, Lng_B):
    if(((float64(Lat_A - Lat_B)) == 0) & ((float64(Lng_A - Lng_B)) == 0)):#防止在统一点坐标
        distance = 0.0
    else:   
        #start = time.clock() 
        ra = 6378.140  # 赤道半径 (km)
        rb = 6356.755  # 极半径 (km)
        flatten = float64((ra - rb) / ra)  # 地球扁率
        rad_lat_A = radians(Lat_A)
        rad_lng_A = radians(Lng_A)
        rad_lat_B = radians(Lat_B)
        rad_lng_B = radians(Lng_B)
        pA = atan(rb / ra * tan(rad_lat_A))
        pB = atan(rb / ra * tan(rad_lat_B))
        xx = float64(acos(sin(pA) * sin(pB)) )+ float64(cos(pA) * cos(pB) * cos(rad_lng_A - rad_lng_B))#防止因为数字过小而报错
        c1 = (sin(xx) - xx) * (sin(pA) + sin(pB)) ** 2 / cos(xx / 2) ** 2
        c2 = (sin(xx) + xx) * (sin(pA) - sin(pB)) ** 2 / sin(xx / 2) ** 2
        dr = flatten / 8 * (c1 - c2)
        distance = ra * (xx + dr)
        
        #end = time.clock()
        #print("The function run time is : %.03f seconds" %(end-start))
    return distance
def getDistanceFromXtoY(lat_a, lng_a, lat_b, lng_b):
    pk = 180 / 3.14169  
    a1 = lat_a / pk  
    a2 = lng_a / pk  
    b1 = lat_b / pk  
    b2 = lng_b / pk  
    t1 = math.cos(a1) * math.cos(a2) * math.cos(b1) * math.cos(b2)  
    t2 = math.cos(a1) * math.sin(a2) * math.cos(b1) * math.sin(b2)  
    t3 = math.sin(a1) * math.sin(b1)  
    t = t1 + t2 + t3
    if t > 1:
        t =1.0
    tt = math.acos(t)  
    return 6366000 * tt    
def calcDistance_1(Lat_A, Lng_A, Lat_B, Lng_B):
    if(((float64(Lat_A - Lat_B)) == 0) & ((float64(Lng_A - Lng_B)) == 0)):#防止在统一点坐标
        distance = 0.0
    else:   
        p1 = cacu.Point()
        p2 = cacu.Point()
        p1.lat = Lat_A
        p1.lng = Lng_A
        p2.lat = Lat_B
        p2.lng = Lat_B
        distance = cacu.getDistance(p1, p2)
    print(distance) 
    return distance
'''
K-NN
'''
def classify(inX,dataSet,labels,k):
    start = time.clock()
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #tile:numpy中的函数。tile将原来的一个数组，扩充成了4个一样的数组。diffMat得到了目标与训练数值之间的差值。
    sqDiffMat   =diffMat**2#各个元素分别平方
    sqDistances =sqDiffMat.sum(axis=1)#对应列相加，即得到了每一个距离的平方
    distances   =sqDistances**0.5#开方，得到距离。
    sortedDistIndicies=distances.argsort()#升序排列
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0) + 1
    #排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    end = time.clock()
   # print("The function run time is : %.03f seconds" %(end-start))
    
    return sortedClassCount[0][0]


def classify_2(inX,dataSet,labels,k):
    start = time.clock()
    distances = zeros((len(dataSet) , 1)) 
    for j in range(len(dataSet)):
        distances[j] =  getDistanceFromXtoY(inX[0],inX[1],dataSet[j][0],dataSet[j][1])#距离
    end = time.clock()
    sortedDistIndicies=distances.argsort()#升序排列
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
   
    print("The function run time is : %.03f seconds" %(end-start))
    
    return sortedClassCount[0][0]

def classify_3(inX,dataSet,labels,k):
    start = time.clock()
    distances = zeros((len(dataSet) , 1)) 
    for j in range(len(dataSet)):
        distances[j] =  cal_dis(inX[0],inX[1],dataSet[j][0],dataSet[j][1])#距离
    sortedDistIndicies=distances.argsort()#升序排列
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        classCount[voteIlabel]=classCount.get(voteIlabel,0)+1
    #排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    end = time.clock()
    print("The function run time is : %.03f seconds" %(end-start))
    
    return sortedClassCount[0][0]

'''
算法评定
'''
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
'''
读取文件转换成矩阵
'''    
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
print('正在初始化参数...')
#2015-11(9:00-24:00) data input
testmat , testclassLabelVector = txt2data('D:\\FILE\\PythonWorkspace\\machineLearning\\data\\buffer.txt')
#testmat = autoNorm(testmat) 使用norm后正确率22%
returnmat , classLabelVector = txt2data('D:\\FILE\\PythonWorkspace\\machineLearning\\data\\8buffer.txt')
#returnmat = autoNorm(returnmat)
#save2txt('D:\\FILE\\PythonWorkspace\\machineLearning\\data\\', testmat, testclassLabelVector)
labels = txt2cata('D:\\FILE\\PythonWorkspace\\machineLearning\\data\\category.txt')
time.sleep(1)
print('矩阵初始化完成！')
#labelSave = txt2cata('D:\\FILE\\PythonWorkspace\\machineLearning\\data\\labelsSave')
#print(labelSave[0])
# start = time.clock()
# for i in range(50000):
#     d = getDistanceFromXtoY(37.480563, 121.467113 , 37.480591  ,121.467926 )
# end = time.clock()
# print(end -start)
# print(d)
# pass
#数据集初始化
#trS , trSL , tS , tSL = builtSet(returnmat, classLabelVector)

pTSL = []
#for i in range(len(tS)):
#    pTSL.append(classify(tS[i] ,trS , trSL , 11))
#    cR = currentRate(pTSL , tSL)
#print(cR)


bar = ProgressBar(total = 100)

print('----------三秒后展示原坐标点图----------')
barProcess(1,3)
plotData(returnmat,classLabelVector , labels)
print('原坐标图展示完成！')

   
num = len(testmat)
print('----------K-NN计算进展情况----------')
for i in range(len(testmat)):
    #if(i == 0 or i == int(len(testmat)/10) or i == int(2*len(testmat)/10) or i == int(3*len(testmat)/10) or  i == int(4*len(testmat)/10) or  i == int(5*len(testmat)/10) or  i == int(6*len(testmat)/10) or  i == int(7*len(testmat)/10) or i == int(8*len(testmat)/10) or  i == int(9*len(testmat)/10) or  i == len(testmat)):
    if(i%(int(len(testmat)/100))==0): 
        bar.log('We have arrived at: ' + str(i + 1))
        bar.move()
    pTSL.append(classify(testmat[i] ,returnmat , classLabelVector , 11))
    
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
print('----------三秒后展示预测坐标点图----------')
barProcess(1,3)
plotData(testmat , pTSL , labels)
print('预测图展示完成！')
print('----------三秒后展示预测情况图----------')
barProcess(1,3)
plotData_2(testmat , pStatus)
print('情况图展示完成！')