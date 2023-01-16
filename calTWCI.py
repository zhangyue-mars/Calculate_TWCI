"""
@author:Zhang Yue
@date  :2023/1/8:15:48
@IDE   :PyCharm
"""
import sys
from osgeo import gdal
import numpy as np
from numpy import pi
from numpy import arctan
import math
import scipy.linalg as la
from tqdm import tqdm

class GRID:
    # 读图像文件
    def read_img(self, filename):
        dataset = gdal.Open(filename)  # 打开文件
        im_width = dataset.RasterXSize  # 栅格矩阵的列数
        im_height = dataset.RasterYSize  # 栅格矩阵的行数
        im_geotrans = dataset.GetGeoTransform()  # 仿射矩阵
        im_proj = dataset.GetProjection()  # 地图投影信息
        im_data = dataset.ReadAsArray(0, 0, im_width, im_height)  # 将数据写成数组，对应栅格矩阵
        im_data = np.array(im_data)
        del dataset  # 关闭对象，文件dataset
        return im_proj, im_geotrans, im_data, im_width, im_height

    # 写文件，写成tiff
    def write_img(self, filename, im_proj, im_geotrans, im_data):
        # 判断栅格数据的数据类型
        if 'int8' in im_data.dtype.name:
            datatype = gdal.GDT_Byte
        elif 'int16' in im_data.dtype.name:
            datatype = gdal.GDT_UInt16
        else:
            datatype = gdal.GDT_Float32
        # 判读数组维数
        if len(im_data.shape) == 3:
            im_bands, im_height, im_width = im_data.shape
        else:
            im_bands, (im_height, im_width) = 1, im_data.shape
        # 创建文件
        driver = gdal.GetDriverByName("GTiff")  # 数据类型必须有，因为要计算需要多大内存空间
        dataset = driver.Create(filename, im_width, im_height, im_bands, datatype)
        dataset.SetGeoTransform(im_geotrans)  # 写入仿射变换参数
        dataset.SetProjection(im_proj)  # 写入投影
        if im_bands == 1:
            dataset.GetRasterBand(1).WriteArray(im_data)  # 写入数组数据
        else:
            for i in range(im_bands):
                dataset.GetRasterBand(i + 1).WriteArray(im_data[i])
        del dataset

#读取每个tiff图像的属性信息
def Readxy(RasterFile):
    ds = gdal.Open(RasterFile,gdal.GA_ReadOnly)
    if ds is None:
        print ('Cannot open ',RasterFile)
        sys.exit(1)
    cols = ds.RasterXSize
    rows = ds.RasterYSize
    band = ds.GetRasterBand(1)
    # data = band.ReadAsArray(0,0,cols,rows)
    noDataValue = band.GetNoDataValue()
    projection=ds.GetProjection()
    geotransform = ds.GetGeoTransform()
    return rows,cols,geotransform,projection,noDataValue

# 根据U10和V10计算风向
def calWindDirection(u, v):
    global WR
    if u <= 0 and v < 0:
        WR = arctan(u / v) * 180 / pi
    elif u <= 0 and v  > 0:
        WR = 180 - arctan(-u / v) * 180 / pi
    elif u >= 0 and v > 0:
        WR = 180 + arctan(u / v) * 180 / pi
    elif u >= 0 and v < 0:
        WR = 360 - arctan(-u / v) * 180 / pi
    elif u < 0 and v == 0:
        WR = 90
    elif u > 0 and v == 0:
        WR = 270
    return WR

def connect(ends):
    d0, d1 = np.abs(np.diff(ends, axis=0))[0]
    if d0 > d1:
        return np.c_[np.linspace(ends[0, 0], ends[1, 0], d0+1, dtype=np.int32),
                     np.round(np.linspace(ends[0, 1], ends[1, 1], d0+1))
                     .astype(np.int32)]
    else:
        return np.c_[np.round(np.linspace(ends[0, 0], ends[1, 0], d1+1))
                     .astype(np.int32),
                     np.linspace(ends[0, 1], ends[1, 1], d1+1, dtype=np.int32)]

def cal_Dist_window_mn(cropped_elev_DEM, x0, y0):
    x = cropped_elev_DEM.shape[0]  # 滑动小窗口行数
    y = cropped_elev_DEM.shape[1]  # 滑动小窗口列数

    dist01 = [[0.0] * x] * y
    dist01 = np.array(dist01)

    for i in range(0, x):
        for j in range(0, y):
            if (i == 0) or (i == x-1) or (j == 0) or (j == y-1):
                start = np.array([i, j])
                end = np.array([x0, y0])
                inputArray = np.vstack((start, end))
                outputArray = connect(inputArray)

                DEMvalue = []
                for it in outputArray:
                    DEMvalue.append([ cropped_elev_DEM[it[0],it[1]], it[0], it[1]])
                MaxIndex = DEMvalue.index(max(DEMvalue))

                for k in range(0, len(DEMvalue)):
                    if k >= MaxIndex:
                        dist01[DEMvalue[k][1], DEMvalue[k][2]] = 1
                    else:
                        dist01[DEMvalue[k][1], DEMvalue[k][2]] = 0
            else:
                continue
    return dist01

# 计算TWI
def calTWI(cropped_W_dist, cropped_elev_DEM, cropped_aspect_DEM, cropped_wild_speed, cropped_wild_Driection, windowsSize,
           diff_elev_max, diff_elev_min, ws_max, ws_min):
    x = cropped_W_dist.shape[0] # 滑动小窗口行数
    y = cropped_W_dist.shape[1] # 滑动小窗口列数
    #
    emptyMatrix = [[0.0] * x] * y  # 空数组，用来存放每个权重值
    W_dist_mn = np.array(emptyMatrix) # 普通数组转换为numpy数组
    #
    x0 = int((x - 1) / 2) # 中心点x坐标
    y0 = int((y - 1) / 2) # 中心点y坐标
    #
    for i in range(0, x):
        for j in range(0, y):
            if (i != x0 and j != y0):
                dis = math.sqrt( ( x0 - i)**2 + (y0 - j)**2 )
                dis2 = dis**(-2)
                W_dist_mn[i, j] =  dis2 # 分子
            else:
                W_dist_mn[i, j] = 0
                continue
    #
    # 求（-2）平方和（分母）
    SUM2 = sum(sum(i) for i in W_dist_mn)
    # print("SUM2:", type(SUM2), SUM2.shape)
    #
    # 计算距离dist权重
    data_W_dist_mn_2 = W_dist_mn / SUM2
    # print("W_dist_mn finish!")
    #
    # for i in range(0, x):
    #     for j in range(0, y):
    #         if (i != x0 and j != y0):
    #             data_W_dist_mn_2[i][j] =  W_dist_mn[i][j] / SUM2
    #         else:
    #             data_W_dist_mn_2[i][j] = 0
    #             continue
    #datatemp2 此时是大W（dist——m,n）计算完毕
    #
    # DEM高程差计算
    DEMtemp = [[0.0] * x] * y  # 存放输出像素值，二维数组
    DEM_W_elve_mn = np.array(DEMtemp)
    #
    for i in range(0, x):
        for j in range(0, y):
            if (i != x0 and j != y0):
                DEM_ElevDist =  cropped_elev_DEM[i, j] - cropped_elev_DEM[x0, y0] # 观察点和周围点的高程差
                if DEM_ElevDist <= 0:
                    DEM_W_elve_mn[i, j] = 0
                else:
                    DEM_W_elve_mn[i, j] = ( DEM_ElevDist - diff_elev_min) / ( diff_elev_max - diff_elev_min)
            else:
                DEM_W_elve_mn[i, j] = 0
                continue
    # W（elev——m，n）DEM权重计算完毕
    # print("W（elev——m，n） finish!")
    #
    ACE_mn = [[0.0] * x] * y  # 存放输出像素值，二维数组
    ACE_mn = np.array(ACE_mn)
    #
    for i in range(0, x):
        for j in range(0, y):
            if (i != x0 and j != y0):
                AngleBetweenTwoVectors = calAspect(i, j, x0, y0, cropped_aspect_DEM[i, j])
                if AngleBetweenTwoVectors < 90:
                    ACE_mn[i, j] = 1 - AngleBetweenTwoVectors / 90
                else:
                    ACE_mn[i, j] = 0
            else:
                ACE_mn[i, j] = 0
                continue
    # print("ACE_mn finish!")
    #
    ACI_mn = ACE_mn * DEM_W_elve_mn
    # print("ACI_mn finish!")
    #
    # 风速 cropped_wild_speed
    W_ws_mn = [[0.0] * x] * y  # 存放输出像素值，二维数组
    W_ws_mn = np.array(W_ws_mn)
    #
    for i in range(0, x):
        for j in range(0, y):
            if (i != x0 and j != y0):
                W_ws_mn[i, j] = (cropped_wild_speed[i, j] - ws_min) / (ws_max - ws_min)
            else:
                W_ws_mn[i, j] = 0
                continue
    # print("W_ws_mn finish!")
    #
    # 风向 cropped_wild_Driection
    WCE_mn = [[0.0] * x] * y  # 存放输出像素值，二维数组
    WCE_mn = np.array(WCE_mn)
    #
    for i in range(0, x):
        for j in range(0, y):
            if (i != x0 and j != y0):
                AngleBetweenTwoVectors = calAspect(i, j, x0, y0, cropped_wild_Driection[i, j])
                if AngleBetweenTwoVectors < 90:
                    WCE_mn[i, j] = 1- AngleBetweenTwoVectors / 90
                else:
                    WCE_mn[i, j] = 0
            else:
                WCE_mn[i, j] = 0
                continue
    # print("WCE_mn finish!")
    #
    WCI_mn = WCE_mn * W_ws_mn
    # print("WCI_mn finish!")
    #
    # 根据DEM计算最大高差距离权重
    Dist_window_mn = cal_Dist_window_mn(cropped_elev_DEM, x0, y0)
    # WCI_mn = WCI_mn_temp * Dist_window_mn
    #
    # 求WCI和ACI的最大值
    MAX_WCI_ACI = [[0.0] * x] * y  # 存放输出像素值，二维数组
    MAX_WCI_ACI = np.array(MAX_WCI_ACI)
    #
    for i in range(0, x):
        for j in range(0, y):
            if (i != x0 and j != y0):
                if(WCI_mn[i][j]>ACI_mn[i, j]):
                    MAX_WCI_ACI[i, j]  = WCI_mn[i, j]
                else:
                    MAX_WCI_ACI[i, j] = ACI_mn[i, j]
            else:
                MAX_WCI_ACI[i, j] = 0
                continue
    #
    MAX_WCI_ACI = MAX_WCI_ACI * Dist_window_mn
    #
    TEMP =  MAX_WCI_ACI * data_W_dist_mn_2
    SUMTEMP = sum(sum(i) for i in TEMP)
    w = windowsSize / 1000
    resultNumber = SUMTEMP / (4 * (w**2))
    # print("TWI finish!", resultNumber.shape)
    #
    return resultNumber

# 计算PBLI
def calPBLI(pblh, pblh_max, pblh_min):
    if pblh >= pblh_max:
        PBLINumber = 0
    else:
        a = pblh - pblh_min
        b = pblh_max - pblh_min
        PBLINumber = 1-(a/b)
    return PBLINumber

# 根据两向量计算角度
# def calAspect(i, j, x0, y0, aspect):
#     a = np.array([i + 10, j + math.tan(aspect) * 10])
#     b = np.array([i, j])
#     c = np.array([x0, y0])
#     ba = a - b
#     bc = c - b
#     cosine_angle = np.dot(ba, bc) / (la.norm(ba) * la.norm(bc))
#     angle = np.arccos(cosine_angle)
#     angle = np.degrees(angle)
#     return abs(angle)
def calAspect(i, j, x0, y0, aspect):
    a = np.array([i + 0, j + 10])
    b = np.array([i, j])
    c = np.array([x0, y0])
    ba = a - b
    bc = c - b
    cosine_angle = np.dot(ba, bc) / (la.norm(ba) * la.norm(bc))
    angle = np.arccos(cosine_angle) # 计算结果为弧度
    angleDegress = np.degrees(angle) # 弧度转换为角度
    angleDifference = abs(aspect - abs(angleDegress)) # 两角度差
    if angleDifference > 180:
        angleDifference = 360 - angleDifference  # angleDifference - 180
    return angleDifference

if __name__ == "__main__":
    print("start")
    run = GRID()
    # 这一个没有参与运算，主要为了读取它的行列数、投影信息、坐标系和noData值
    rows, cols, geotransform, projection, noDataValue = Readxy('E:\\RemoteSensing\\TWCI_XYKH20\\Preprocesseddata1\\UVwindDriection.tif')
    print(rows, cols, geotransform, projection, noDataValue)
    #
    # 计算风向
    # _, _, datau10, _, _ = run.read_img("E:\\RemoteSensing\\Preprocesseddata1\\u10Mean.tif")
    # dataArrayU10 = np.array(datau10)
    # print(type(dataArrayU10))
    # _, _, datav10, _, _ = run.read_img("E:\\RemoteSensing\\Preprocesseddata1\\v10Mean.tif")
    # dataArrayV10 = np.array(datav10)
    # print(type(dataArrayV10))
    #
    # # 写数据
    # average = [[0.0] * cols] * rows
    # average = np.array(average)
    #
    # for i in range(0, rows):
    #     for j in range(0, cols):
    #         # print(noDataValue, count)
    #         if (dataArrayU10[i, j] == noDataValue):  # 处理图像中的noData
    #             average[i, j] = -9999
    #         else:
    #             u_value = dataArrayU10[i, j]
    #             v_value = dataArrayV10[i, j]
    #             average[i, j] = calWindDirection(u_value, v_value)  # 求风向
    # 保存风向数据
    # run.write_img('E:\\RemoteSensing\\resultData' + '//' + 'WindDirection' + '.tif', projection, geotransform, average)
    #
    # 计算风速开始
    # _, _, datau10, _, _ = run.read_img("E:\\RemoteSensing\\Preprocesseddata1\\u10Mean.tif")# 计算风速
    # dataArrayU10 = np.array(datau10)
    # # print(dataArrayU10.shape)
    # _, _, datav10, _, _ = run.read_img("E:\\RemoteSensing\\Preprocesseddata1\\v10Mean.tif")
    # dataArrayV10 = np.array(datav10)
    # # print(dataArrayV10.shape)
    # #
    # # 写数据
    # wild_speed = [[0.0] * cols] * rows  # 存放平均值，二维数组
    # wild_speed = np.array(wild_speed)
    # #
    # for i in range(0, rows):
    #     for j in range(0, cols):
    #         # print(noDataValue, count)
    #         if (dataArrayU10[i, j] == noDataValue):  # 处理图像中的noData
    #             wild_speed[i, j] = -9999
    #         else:
    #             u_value = dataArrayU10[i, j]
    #             v_value = dataArrayV10[i, j]
    #             wild_speed[i, j] = math.sqrt((u_value**2) + (v_value**2))  # 求风速
    # 计算风速数据结束
    #
    # 读取风向数据
    _, _, si10Mean, _, _ = run.read_img("E:\\RemoteSensing\\TWCI_XYKH20\\Preprocesseddata1\\si10Mean.tif")
    wild_speed = np.array(si10Mean)
    # print(type(dataArray_UV_WindDirection))
    #
    # 不能删
    data = [[0.0] * cols] * rows  # 存放输出像素值，二维数组,用来保存结果
    twciArray = np.array(data)
    # x，y为data的行列数
    x = twciArray.shape[0]
    y = twciArray.shape[1]
    # 窗口尺寸，单位KM
    windowsSize = 1200 # Unit:KM
    # 滑动窗口尺寸，窗口尺寸除以像元分辨率
    CropSize = 1200 / 5
    # 补0行数
    addZoreNum = int(CropSize / 2)
    # Numpy为图片四周补0
    # padimg = np.pad(dataArray_UV_WindDirection, ((addZoreNum, addZoreNum), (addZoreNum, addZoreNum)), 'constant', constant_values=(0, 0))
    # print(padimg.shape)
    # padimg_UV_WindDirection = padimg
    #
    # 读取DEM高程数据
    _, _, DEM, _, _ = run.read_img("E:\\RemoteSensing\\TWCI_XYKH20\\Preprocesseddata1\\NCPDEM.tif")
    DEM = np.array(DEM)
    DEM = np.pad(DEM, ((addZoreNum, addZoreNum), (addZoreNum, addZoreNum)), 'constant',
                            constant_values=(0, 0))
    #
    # 读取坡向数据
    _, _, aspect, _, _ = run.read_img("E:\\RemoteSensing\\TWCI_XYKH20\\Preprocesseddata1\\NCPaspect.tif")
    aspect = np.array(aspect)
    aspect = np.pad(aspect, ((addZoreNum, addZoreNum), (addZoreNum, addZoreNum)), 'constant',
                            constant_values=(0, 0))
    #
    # 读取UV计算的风向数据
    _, _, wild_Driection, _, _ = run.read_img("E:\\RemoteSensing\\TWCI_XYKH20\\Preprocesseddata1\\UVwindDriection.tif")
    wild_Driection = np.array(wild_Driection)
    wild_Driection = np.pad(wild_Driection, ((addZoreNum, addZoreNum), (addZoreNum, addZoreNum)), 'constant',
                            constant_values=(0, 0))
    #
    # 读取风速数据，由uv计算得来，后面考虑用si10替换
    wild_speed = np.pad(wild_speed, ((addZoreNum, addZoreNum), (addZoreNum, addZoreNum)), 'constant',
                            constant_values=(0, 0))
    #
    # 读取blh数据
    _, _, PBLH, _, _ = run.read_img("E:\\RemoteSensing\\TWCI_XYKH20\\Preprocesseddata1\\blhMean.tif")
    PBLH = np.array(PBLH)
    PBLH = np.pad(PBLH, ((addZoreNum, addZoreNum), (addZoreNum, addZoreNum)), 'constant',
                            constant_values=(0, 0))
    #
    # blh的最大值和最小值，单位m
    pblh_max = 2000
    pblh_min = 0
    # 高程差的最大最小值，单位m
    diff_elev_max = 2000
    diff_elev_min = 0
    #风速的最大最小值,单位m/s
    ws_max = 5
    ws_min = 0
    #
    # 遍历每个像素
    for i in tqdm(range(700, 701)): # 测试用
        for j in range(1150, 1151): # 测试用
    # for i in tqdm(range(addZoreNum, x + addZoreNum)):
    #     for j in range(addZoreNum, y + addZoreNum):
            if (wild_Driection[i, j] == noDataValue or DEM[i, j] == noDataValue ):  # 处理图像中的noData
                twciArray[i, j] = -9999
            else:
                # 不同变量的滑动窗口
                cropped_W_dist = wild_Driection[
                                    int(i - CropSize / 2): int((i + CropSize / 2) + 1),
                                    int(j - CropSize / 2): int((j + CropSize / 2) + 1)]
                #
                cropped_elev_DEM = DEM[
                                    int(i - CropSize / 2): int((i + CropSize / 2) + 1),
                                    int(j - CropSize / 2): int((j + CropSize / 2) + 1)]
                #
                cropped_aspect_DEM = aspect[
                                    int(i - CropSize / 2): int((i + CropSize / 2) + 1),
                                    int(j - CropSize / 2): int((j + CropSize / 2) + 1)]
                #
                cropped_wild_speed = wild_speed[
                                    int(i - CropSize / 2): int((i + CropSize / 2) + 1),
                                    int(j - CropSize / 2): int((j + CropSize / 2) + 1)]
                #
                cropped_wild_Driection = wild_Driection[
                                    int(i - CropSize / 2): int((i + CropSize / 2) + 1),
                                    int(j - CropSize / 2): int((j + CropSize / 2) + 1)]
                #
                # 计算每个像素的TWI
                TWIValue = calTWI(cropped_W_dist, cropped_elev_DEM, cropped_aspect_DEM, cropped_wild_speed, cropped_wild_Driection, windowsSize,
                                  diff_elev_max, diff_elev_min, ws_max, ws_min)
                # print("TWIValue", TWIValue.shape)
                #
                # 计算PBLI
                PBLHValue = calPBLI(PBLH[i, j], pblh_max, pblh_min)
                # print("PBLHValue", PBLHValue.shape)
                # print(TWIValue, PBLHValue)
                #
                # 计算TWCI
                TWCIVale = TWIValue * PBLHValue
                print("TWCIVale:", TWCIVale)
                #
                # 计算结束，将TWCI保存到每个像素
                twciArray[i, j] = TWCIVale
                #
    # 保存TWCI结果为tiff
    run.write_img('E:\\RemoteSensing\\TWCI_XYKH20\\resultData' + '//' + 'TWCI' + '.tif', projection, geotransform, twciArray)
    print("TWCI finish!")