import numpy as np
from PIL import Image
import math

k_x =1
k_y =1
# h = 2240
# w = 3069
h = 2240
w = 3069


height = int(h*k_x)
width = int(w*k_y)
range_fanwei = math.ceil(height*width/40000)

depth = 3

range_fanwei_te002=112

range_fanwei_te002_all=171

range_fanwei_te004=328

range_fanwei_nvhai=534
# 创建全为0的矩阵
matrix = np.zeros((height, width, depth))

for m in range(range_fanwei-1):
    inpath='D:/zyw/XUNLIAN/output/y_pred_'+str(m)+'.txt'
    data = np.loadtxt(inpath)
    for n in range(40000):
        num= m*40000+n
        i = num // width
        j = num % width
        matrix[i, j, 0] = data[n, 0]
        matrix[i, j, 1] = data[n, 1]
        matrix[i, j, 2] = data[n, 2]


matrix = matrix.astype(np.uint8)

# 将矩阵转换为图像
image = Image.fromarray(matrix)

# # 保存图像为BMP文件
# image.save("D:/zyw/XUNLIAN/tu/fruitmdn.bmp")
# 保存图像为BMP文件
image.save("D:/zyw/XUNLIAN/tu/xgb0730.tif")


print("矩阵已保存为BMP文件")
