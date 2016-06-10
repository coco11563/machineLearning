'''
Created on 2016年6月8日

@author: coco1
'''
import math
import time
class Point:  
    pass  
  
def max(a,b):  
    if a>b:  
        return a  
    return b  
def min(a,c):  
    if a>c:  
        return c  
    return a  
  
def lw(a, b, c):  
#     b != n && (a = Math.max(a, b));  
#     c != n && (a = Math.min(a, c));  
    a = max(a,b)  
    a = min(a,c)  
    return a  
  
def ew(a, b, c):  
      
    while a > c:  
        a -= c - b  
    while a < b:  
        a += c - b  
    return a  
          
  
def oi(a):  
    return math.pi * a / 180  
  
def Td(a, b, c, d):   
    return 6370996.81 * math.acos(math.sin(c) * math.sin(d) + math.cos(c) * math.cos(d) * math.cos(b - a))  
  
def Wv(a, b):  
    if not a or not b:   
        return 0;  
    a.lng = ew(a.lng, -180, 180);  
    a.lat = lw(a.lat, -74, 74);  
    b.lng = ew(b.lng, -180, 180);  
    b.lat = lw(b.lat, -74, 74);  
    return Td(oi(a.lng), oi(b.lng), oi(a.lat), oi(b.lat))  
  
def getDistance(a, b):  
    c = Wv(a, b);  
    return c
def getDistanceFromXtoY(lat_a, lng_a, lat_b, lng_b):
    pk = 180 / 3.14169  
    a1 = lat_a / pk  
    a2 = lng_a / pk  
    b1 = lat_b / pk  
    b2 = lng_b / pk  
    t1 = math.cos(a1) * math.cos(a2) * math.cos(b1) * math.cos(b2)  
    t2 = math.cos(a1) * math.sin(a2) * math.cos(b1) * math.sin(b2)  
    t3 = math.sin(a1) * math.sin(b1)  
    tt = math.acos(t1 + t2 + t3)  
    return 6366000 * tt    
# p1 = Point()  
# p1.lat = 37.480563  
# p1.lng = 121.467113  
# p2 = Point()  
# p2.lat = 37.480591  
# p2.lng = 121.467926  
# start = time.clock()
# for i in range(50000):
#     d = getDistance(p1, p2)
# end = time.clock()
# print(d)
# print(end -start)
# start2 = time.clock()
# for i in range(50000): 
#     d2 = getDistanceFromXtoY(p1.lat,p1.lng, p2.lat,p2.lng)
# print(d2) 
# end2 = time.clock()
# print(end2 - start2)