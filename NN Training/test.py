# 取第一个Point进行预测
point = 5675 #116.60 40.05

# point与经纬度转化关系
# 北京市经纬度边界
# bounds = [115.7, 39.4, 117.4, 41.6]
bounds = [116.0, 39.6, 116.8, 40.2]
MaxX = bounds[2]
MinX = bounds[0]
MaxY = bounds[3]
MinY = bounds[1]
# 坐标采样个数,即Sample_Num*Sample_Num
Sample_Num = 100
deltaX = (MaxX-MinX)/Sample_Num
deltaY = (MaxY-MinY)/Sample_Num

point_X = (point % 100) * deltaX + MinX
point_Y = (point / 100) * deltaX + MinY
print(point_X, point_Y)