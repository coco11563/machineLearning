# -*- coding: utf-8 -*-
'''
Created on 2016年7月9日

@author: Shaow
'''
import matplotlib.pyplot as plt
from matplotlib.pyplot import axis, spectral    
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
