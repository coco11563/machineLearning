# -*- coding: utf-8 -*-
'''
Created on 2016年7月9日

@author: Shaow
'''
from numpy import *
import sys,time  
from _ctypes import Array
correctmatrix = zeros((10,2),double)
correctmatrix[:,1] = 1;
decision = zeros((10,2),double)
print(decision)
decision = decision - correctmatrix
print(decision)
d = array([[1,0],[0,-1]])
print(d)
c = dot(decision,d) 
print(c)
    