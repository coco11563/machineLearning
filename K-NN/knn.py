
# coding = UTF-8
'''
Created on 2016年6月1日
@author: coco1
'''
from numpy import* 
import operator
from numpy.linalg.linalg import solve
from math import *
import time
import sys
from buildSet import *
from buildSet import autoNorm





                


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
    print(classCount)
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
    end = time.clock()
   # print("The function run time is : %.03f seconds" %(end-start))
    
    return sortedClassCount[0][0]

def classifyByPoi(inX,dataSet,labels,num,k):#{num,dis}
    decision = zeros((k,2),double)
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #tile:numpy中的函数。tile将原来的一个数组，扩充成了4个一样的数组。diffMat得到了目标与训练数值之间的差值。
    sqDiffMat   =diffMat**2#各个元素分别平方
    sqDistances =sqDiffMat.sum(axis=1)#对应列相加，即得到了每一个距离的平方
    distances   =sqDistances**0.5#开方，得到距离。
    sortedDistIndicies=distances.argsort()#升序排列
    
    classCount={}
    voteIlabel = []
    for i in range(k):
        voteIlabel.append(labels[sortedDistIndicies[i]])
        checkinNum = num[sortedDistIndicies[i]]
        dis = distances[sortedDistIndicies[i]]
        decision[i,0] = checkinNum
        decision[i,1] = dis
        
        #classCount[voteIlabel]=classCount.get(voteIlabel,0) + 1
        #sum([[0, 1], [0, 5]], axis=1)    #axis=1 是按行求和
    decision = autoNorm(decision)
    decision = decision* [0.7,0.3] #[修改这部分更改权重]
    decisionMatrix = decision.sum(axis = 1)
    decisionIndicies = decisionMatrix.argsort()     
    #排序

   # print("The function run time is : %.03f seconds" %(end-start))
    return voteIlabel[decisionIndicies[1]]
def classifyByPoi_1(inX,dataSet,labels,num,k):
    dataSetSize = dataSet.shape[0]
    diffMat = tile(inX,(dataSetSize,1)) - dataSet #tile:numpy中的函数。tile将原来的一个数组，扩充成了4个一样的数组。diffMat得到了目标与训练数值之间的差值。
    sqDiffMat   =diffMat**2#各个元素分别平方
    sqDistances =sqDiffMat.sum(axis=1)#对应列相加，即得到了每一个距离的平方
    distances   =sqDistances**0.5#开方，得到距离。
    sortedDistIndicies=distances.argsort()#升序排列
    classCount={}
    for i in range(k):
        voteIlabel=labels[sortedDistIndicies[i]]
        checkinNum = num[sortedDistIndicies[i]]
        dis = distances[sortedDistIndicies[i]]
        #classCount[voteIlabel]=classCount.get(voteIlabel,0) + 1
        classCount[voteIlabel]=classCount.get(voteIlabel,0) + int(checkinNum)
    print(classCount)   
    #排序
    sortedClassCount=sorted(classCount.items(),key=operator.itemgetter(1),reverse=True)
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

      

