'''
Created on 2016年6月6日
@author: coco1
'''
import math
def LL2UTM_USGS(a, f, lat, lon, lonOrigin, FN):
    '''
    a = 6378136.49m
    b = 6356755.00m
    lonOrigin = 114.17
    FN = 0    
    ** Input：(a, f, lat, lon, lonOrigin, FN)
    ** a 椭球体长半轴
    ** f 椭球体扁率 f=(a-b)/a 其中b代表椭球体的短半轴
    ** lat 经过UTM投影之前的纬度
    ** lon 经过UTM投影之前的经度
    ** lonOrigin 中央经度线
    ** FN 纬度起始点，北半球为0，南半球为10000000.0m
    ---------------------------------------------
    ** Output:(UTMNorthing, UTMEasting)
    ** UTMNorthing 经过UTM投影后的纬度方向的坐标
    ** UTMEasting 经过UTM投影后的经度方向的坐标
    ---------------------------------------------
    ** 功能描述：UTM投影
    ** 作者： Ace Strong
    ** 单位： CCA NUAA
    ** 创建日期：2008年7月19日
    ** 版本：1.0
    ** 本程序实现的公式请参考
    ** "Coordinate Conversions and Transformations including Formulas" p35.
    ** & http://www.uwgb.edu/dutchs/UsefulData/UTMFormulas.htm
    '''
    # e表示WGS84第一偏心率,eSquare表示e的平方
    eSquare = 2*f - f*f
    k0 = 0.9996
    # 确保longtitude位于-180.00----179.9之间
    lonTemp = (lon+180)-int((lon+180)/360)*360-180
    latRad = math.radians(lat)
    lonRad = math.radians(lonTemp)
    lonOriginRad = math.radians(lonOrigin)
    e2Square = (eSquare)/(1-eSquare)
    V = a/math.sqrt(1-eSquare*math.sin(latRad)**2)
    T = math.tan(latRad)**2
    C = e2Square*math.cos(latRad)**2
    A = math.cos(latRad)*(lonRad-lonOriginRad)
    M = a*((1-eSquare/4-3*eSquare**2/64-5*eSquare**3/256)*latRad
    -(3*eSquare/8+3*eSquare**2/32+45*eSquare**3/1024)*math.sin(2*latRad)
    +(15*eSquare**2/256+45*eSquare**3/1024)*math.sin(4*latRad)
    -(35*eSquare**3/3072)*math.sin(6*latRad))
    # x
    UTMEasting = k0*V*(A+(1-T+C)*A**3/6
    + (5-18*T+T**2+72*C-58*e2Square)*A**5/120)+ 500000.0
    # y
    UTMNorthing = k0*(M+V*math.tan(latRad)*(A**2/2+(5-T+9*C+4*C**2)*A**4/24
    +(61-58*T+T**2+600*C-330*e2Square)*A**6/720))
    # 南半球纬度起点为10000000.0m
    UTMNorthing += FN
    return (UTMEasting,UTMNorthing)


e , n = LL2UTM_USGS(6378136.49,6356755.00,30.45821,114.272369,114.17,0)
print(e , n)