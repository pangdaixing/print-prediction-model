import cv2
import numpy as np
from PIL import Image
#将矩阵保存到excel里
import pandas as pd
import multiprocessing as mp
# 修改允许的最大像素数
Image.MAX_IMAGE_PIXELS = None

def kuobaibian(img,border_width):

    # 获取图像的宽和高
    width, height = img.size

    # 定义白色
    white = (0, 0, 0, 0)


    # 给图像加一圈白边
    img_with_border = Image.new(img.mode, (width + border_width * 2, height + border_width * 2), white)
    img_with_border.paste(img, (border_width, border_width))


    return img_with_border

def actual_x_create(height):
    actual_x = np.zeros((height, 1))
    for i in range(height):
        actual_x[i][0] = i



    return actual_x

def yzeng(yjuzhen,border_width):

    for i in range(border_width):
        yjuzhen = np.insert(yjuzhen, 0, 0, axis=1)
        yjuzhen = np.column_stack((yjuzhen, 0))
    return yjuzhen

def xzeng(xjuzhen,border_width):

    for i in range(border_width):
        xjuzhen = np.insert(xjuzhen, 0, -1-i, axis=0)
        xjuzhen = np.row_stack((xjuzhen, len(xjuzhen)-1-i))
    return xjuzhen




def create_data_size(img,img2,xfan,yfan,k_x,k_y,block_size_ro,block_size_co,pro,pco,save_path,path_pentou,mode = 0):
    #用于处理更大尺寸的数据集  实验用

    if mode == 0:
        real_height = img.height
        real_width = img.width

        # 加一圈白边
        border_width = 4
        img_with_border = kuobaibian(img, border_width)

        # 读取长宽
        width, height = img_with_border.size

        img = np.array(img_with_border)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k

        img2 = np.array(img2)
        r, g, b = img2[..., 0], img2[..., 1], img2[..., 2]
        img_r = r
        img_g = g
        img_b = b

        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        actual_offset_y_c = yzeng(actual_offset_y_c, border_width)

        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8
        actual_offset_y_m = yzeng(actual_offset_y_m, border_width)

        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        actual_offset_y_y = yzeng(actual_offset_y_y, border_width)

        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        actual_offset_y_k = yzeng(actual_offset_y_k, border_width)

        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8
        actual_offset_x_c = yzeng(actual_offset_x_c, border_width)

        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8
        actual_offset_x_m = yzeng(actual_offset_x_m, border_width)

        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8
        actual_offset_x_y = yzeng(actual_offset_x_y, border_width)

        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8
        actual_offset_x_k = yzeng(actual_offset_x_k, border_width)

        # actual_offset_x_c = np.zeros((1, actual_offset_x_c.shape[1]))
        # actual_offset_x_m = np.zeros((1, actual_offset_x_c.shape[1]))
        # actual_offset_x_y = np.zeros((1, actual_offset_x_c.shape[1]))
        # actual_offset_x_k = np.zeros((1, actual_offset_x_c.shape[1]))

        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)
        actual_D_c = np.transpose(actual_D_c) / 16
        actual_D_c = yzeng(actual_D_c, border_width)

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)
        actual_D_m = np.transpose(actual_D_m) / 16
        actual_D_m = yzeng(actual_D_m, border_width)

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)
        actual_D_y = np.transpose(actual_D_y) / 16
        actual_D_y = yzeng(actual_D_y, border_width)

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)
        actual_D_k = np.transpose(actual_D_k) / 16
        actual_D_k = yzeng(actual_D_k, border_width)

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12 + 3

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))
        # 生成随机数组
        random_x = np.random.randint(1, real_height - pro, size=(1, xfan * yfan))
        random_y = np.random.randint(1, real_width - pco, size=(1, xfan * yfan))

        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                i = random_x[0][m * yfan + n]
                j = random_y[0][m * yfan + n]

                index_data_x = m * yfan + n

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                            block_size_co / 2))

                        i_s = int(round(i / k_x)) + a
                        j_s = int(round(j / k_y)) + b

                        index_y_c = j_s - 2*int(j_s/2048)
                        index_i_c = i_s

                        index_y_m = j_s - 2*int(j_s/2048)
                        index_i_m = i_s

                        index_y_y = j_s - 2*int(j_s/2048)
                        index_i_y = i_s

                        index_y_k = j_s - 3*int(j_s/2048)
                        index_i_k = i_s

                        # c
                        if (img_c[index_i_c + border_width][index_y_c + border_width] != 0):
                            data_cpt[index_data_x][index_date_y] = k_x * index_i_c + actual_offset_x_c[0][
                                index_y_c + border_width] - i
                            data_cpt[index_data_x][index_date_y + 1] = k_y * index_y_c + \
                                                                       actual_offset_y_c[0][
                                                                           index_y_c + border_width] - j
                            data_cpt[index_data_x][index_date_y + 2] = \
                                actual_D_c[0][
                                    index_y_c + border_width]

                        # m
                        if (img_m[index_i_m + border_width][index_y_m + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 3] = k_x * index_i_m + \
                                                                                    actual_offset_x_m[0][
                                                                                        index_y_m + border_width] - i
                        data_cpt[index_data_x][index_date_y + xiaochicun * 3 + 1] = k_y * index_y_m + \
                                                                                    actual_offset_y_m[0][
                                                                                        index_y_m + border_width] - j
                        data_cpt[index_data_x][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][
                            index_y_m + border_width]

                        # y

                        if (img_y[index_i_y + border_width][index_y_y + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 6] = k_x * index_i_y + \
                                                                                    actual_offset_x_y[0][
                                                                                        index_y_y + border_width] - i
                        data_cpt[index_data_x][index_date_y + xiaochicun * 6 + 1] = k_y * index_y_y + \
                                                                                    actual_offset_y_y[0][
                                                                                        index_y_y + border_width] - j
                        data_cpt[index_data_x][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][
                            index_y_y + border_width]

                        # k

                        if (img_k[index_i_k + border_width][index_y_k + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9] = k_x * index_i_k + \
                                                                                    actual_offset_x_k[0][
                                                                                        index_y_k + border_width] - i
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9 + 1] = k_y * index_y_k + \
                                                                                        actual_offset_y_k[0][
                                                                                            index_y_k + border_width] - j
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][
                                index_y_k + border_width]

                data_cpt[m * yfan + n][xiaochicun * 12] = img_r[i + pro][j + pco]
                data_cpt[m * yfan + n][xiaochicun * 12 + 1] = img_g[i + pro][j + pco]
                data_cpt[m * yfan + n][xiaochicun * 12 + 2] = img_b[i + pro][j + pco]

        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

    if mode == 1:
        # 加一圈白边

        img_width, img_height = img.size

        xfan = int(img_height*k_x)
        yfan = int(img_width*k_y)


        border_width = 4
        img_with_border = kuobaibian(img, border_width)


        img = np.array(img_with_border)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k


        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        actual_offset_y_c = yzeng(actual_offset_y_c, border_width)

        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8
        actual_offset_y_m = yzeng(actual_offset_y_m, border_width)

        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        actual_offset_y_y = yzeng(actual_offset_y_y, border_width)

        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        actual_offset_y_k = yzeng(actual_offset_y_k, border_width)

        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8
        actual_offset_x_c = yzeng(actual_offset_x_c, border_width)

        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8
        actual_offset_x_m = yzeng(actual_offset_x_m, border_width)

        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8
        actual_offset_x_y = yzeng(actual_offset_x_y, border_width)

        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8
        actual_offset_x_k = yzeng(actual_offset_x_k, border_width)



        # 创建实际大小D(半径)
        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)
        actual_D_c = np.transpose(actual_D_c) / 16
        actual_D_c = yzeng(actual_D_c, border_width)

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)
        actual_D_m = np.transpose(actual_D_m) / 16
        actual_D_m = yzeng(actual_D_m, border_width)

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)
        actual_D_y = np.transpose(actual_D_y) / 16
        actual_D_y = yzeng(actual_D_y, border_width)

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)
        actual_D_k = np.transpose(actual_D_k) / 16
        actual_D_k = yzeng(actual_D_k, border_width)

        xiaochicun = block_size_ro * block_size_co
        chicun = xiaochicun * 12

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun + 3))


        for m in range(0, xfan):
            for n in range(0, yfan):
                i = m
                j = n

                index_data_x = m * yfan + n

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                            block_size_co / 2))

                        i_s = int(round(i / k_x)) + a
                        j_s = int(round(j / k_y)) + b

                        index_y_c = j_s - 2*int(j_s/2048)
                        index_i_c = i_s

                        index_y_m = j_s - 2*int(j_s/2048)
                        index_i_m = i_s

                        index_y_y = j_s - 2*int(j_s/2048)
                        index_i_y = i_s

                        index_y_k = j_s - 3*int(j_s/2048)
                        index_i_k = i_s

                        # c
                        if (img_c[index_i_c + border_width][index_y_c + border_width] != 0):
                            data_cpt[index_data_x][index_date_y] = k_x * index_i_c + actual_offset_x_c[0][
                                index_y_c + border_width] - i
                            data_cpt[index_data_x][index_date_y + 1] = k_y * index_y_c + \
                                                                       actual_offset_y_c[0][
                                                                           index_y_c + border_width] - j
                            data_cpt[index_data_x][index_date_y + 2] = \
                                actual_D_c[0][
                                    index_y_c + border_width]

                        # m
                        if (img_m[index_i_m + border_width][index_y_m + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 3] = k_x * index_i_m + \
                                                                                    actual_offset_x_m[0][
                                                                                        index_y_m + border_width] - i
                        data_cpt[index_data_x][index_date_y + xiaochicun * 3 + 1] = k_y * index_y_m + \
                                                                                    actual_offset_y_m[0][
                                                                                        index_y_m + border_width] - j
                        data_cpt[index_data_x][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][
                            index_y_m + border_width]

                        # y

                        if (img_y[index_i_y + border_width][index_y_y + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 6] = k_x * index_i_y + \
                                                                                    actual_offset_x_y[0][
                                                                                        index_y_y + border_width] - i
                        data_cpt[index_data_x][index_date_y + xiaochicun * 6 + 1] = k_y * index_y_y + \
                                                                                    actual_offset_y_y[0][
                                                                                        index_y_y + border_width] - j
                        data_cpt[index_data_x][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][
                            index_y_y + border_width]

                        # k

                        if (img_k[index_i_k + border_width][index_y_k + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9] = k_x * index_i_k + \
                                                                                    actual_offset_x_k[0][
                                                                                        index_y_k + border_width] - i
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9 + 1] = k_y * index_y_k + \
                                                                                        actual_offset_y_k[0][
                                                                                            index_y_k + border_width] - j
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][
                                index_y_k + border_width]



        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)



def create_data_1015(img,img2,xfan,yfan,k_x,k_y,block_size_ro,block_size_co,pro,pco,save_path,path_pentou,mode = 0):
    #用于处理更大尺寸的数据集  实验用

    if mode ==0:
        real_height = img2.height
        real_width = img2.width

        # 加一圈白边
        border_width = 10
        img = kuobaibian(img, border_width)

        img = np.array(img)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k

        img2 = np.array(img2)
        r, g, b = img2[..., 0], img2[..., 1], img2[..., 2]
        img_r = r
        img_g = g
        img_b = b

        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        #实际位置
        actual_position_y_c = np.arange(0, actual_offset_y_c.shape[1])+actual_offset_y_c
        #利用 argsort 来获取排序后的索引
        indices_y_c  = np.argsort(actual_position_y_c)



        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8

        #实际位置
        actual_position_y_m = np.arange(0, actual_offset_y_m.shape[1])+actual_offset_y_m
        #利用 argsort 来获取排序后的索引
        indices_y_m  = np.argsort(actual_position_y_m)


        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        #实际位置
        actual_position_y_y = np.arange(0, actual_offset_y_y.shape[1])+actual_offset_y_y
        #利用 argsort 来获取排序后的索引
        indices_y_y  = np.argsort(actual_position_y_y)


        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        #实际位置
        actual_position_y_k = np.arange(0, actual_offset_y_k.shape[1])+actual_offset_y_k
        #利用 argsort 来获取排序后的索引
        indices_y_k  = np.argsort(actual_position_y_k)


        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8


        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8


        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8


        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8



        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)

        actual_D_c = np.transpose(actual_D_c) / 16

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)

        actual_D_m = np.transpose(actual_D_m) / 16

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)

        actual_D_y = np.transpose(actual_D_y) / 16

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)

        actual_D_k = np.transpose(actual_D_k) / 16

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12 + 3

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))
        # 生成随机数组
        random_x = np.random.randint(1, real_height - pro, size=(1, xfan * yfan))
        random_y = np.random.randint(1, real_width - pco, size=(1, xfan * yfan))

        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                i = random_x[0][m * yfan + n]
                j = random_y[0][m * yfan + n]

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):
                        i_s = i + a
                        j_s =j + b

                        index_i_c = int(i_s/k_x - actual_offset_x_c[0][indices_y_c[0][int(j_s/k_y)]])
                        index_y_c = indices_y_c[0][int(j_s/k_y)]

                        index_i_m = int(i_s/k_x - actual_offset_x_m[0][indices_y_m[0][int(j_s/k_y)]])
                        index_y_m = indices_y_m[0][int(j_s/k_y)]

                        index_i_y = int(i_s/k_x - actual_offset_x_y[0][indices_y_y[0][int(j_s/k_y)]])
                        index_y_y = indices_y_y[0][int(j_s/k_y)]

                        index_i_k = int(i_s/k_x - actual_offset_x_k[0][indices_y_k[0][int(j_s/k_y)]])
                        index_y_k = indices_y_k[0][int(j_s/k_y)]


                        if i_s >= 0 and j_s>=0:
                            if (img_c[index_i_c+border_width][index_y_c+border_width] != 0):
                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))] = k_x * index_i_c + k_x * actual_offset_x_c[0][index_y_c] - i

                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2)) + 1] = k_y * actual_position_y_c[0][index_y_c]-j

                                data_cpt[m * yfan + n][
                                    3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                        block_size_co / 2)) + 2] = actual_D_c[0][index_y_c]

                            if (img_m[index_i_m+border_width][index_y_m+border_width] != 0):
                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2)) + xiaochicun * 3] = k_x * index_i_m + k_x * actual_offset_x_m[0][index_y_m] - i

                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2)) + xiaochicun * 3 + 1] = k_y * actual_position_y_m[0][index_y_m]-j

                                data_cpt[m * yfan + n][
                                    3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                        block_size_co / 2))+ xiaochicun * 3 + 2] = actual_D_m[0][index_y_m]

                            if (img_y[index_i_y+border_width][index_y_y+border_width] != 0):
                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))+ xiaochicun * 6] = k_x * index_i_y + k_x * actual_offset_x_y[0][index_y_y] - i

                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))+ xiaochicun * 6 + 1] = k_y * actual_position_y_y[0][index_y_y]-j

                                data_cpt[m * yfan + n][
                                    3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                        block_size_co / 2))+ xiaochicun * 6 + 2] = actual_D_y[0][index_y_y]

                            if (img_k[index_i_k+border_width][index_y_k+border_width] != 0):
                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))+ xiaochicun * 9] = k_x * index_i_k + k_x * actual_offset_x_k[0][index_y_k] - i

                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2)) + xiaochicun * 9 + 1] = k_y * actual_position_y_k[0][index_y_k]-j

                                data_cpt[m * yfan + n][
                                    3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                        block_size_co / 2)) + xiaochicun * 9 + 2] = actual_D_k[0][index_y_k]


                data_cpt[m * yfan + n][xiaochicun * 12] = img_r[i + pro][j + pco]
                data_cpt[m * yfan + n][xiaochicun * 12 + 1] = img_g[i + pro][j + pco]
                data_cpt[m * yfan + n][xiaochicun * 12 + 2] = img_b[i + pro][j + pco]

        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

    if mode ==1:

        xfan = int(img.height*k_x)
        yfan = int(img.width*k_y)

        # 加一圈白边
        border_width = 10
        img = kuobaibian(img, border_width)

        img = np.array(img)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k


        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        #实际位置
        actual_position_y_c = np.arange(0, actual_offset_y_c.shape[1])+actual_offset_y_c
        #利用 argsort 来获取排序后的索引
        indices_y_c  = np.argsort(actual_position_y_c)



        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8

        #实际位置
        actual_position_y_m = np.arange(0, actual_offset_y_m.shape[1])+actual_offset_y_m
        #利用 argsort 来获取排序后的索引
        indices_y_m  = np.argsort(actual_position_y_m)


        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        #实际位置
        actual_position_y_y = np.arange(0, actual_offset_y_y.shape[1])+actual_offset_y_y
        #利用 argsort 来获取排序后的索引
        indices_y_y  = np.argsort(actual_position_y_y)


        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        #实际位置
        actual_position_y_k = np.arange(0, actual_offset_y_k.shape[1])+actual_offset_y_k
        #利用 argsort 来获取排序后的索引
        indices_y_k  = np.argsort(actual_position_y_k)


        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8


        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8


        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8


        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8



        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)

        actual_D_c = np.transpose(actual_D_c) / 16

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)

        actual_D_m = np.transpose(actual_D_m) / 16

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)

        actual_D_y = np.transpose(actual_D_y) / 16

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)

        actual_D_k = np.transpose(actual_D_k) / 16

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))


        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                i = m
                j = n

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):
                        i_s = i + a
                        j_s =j + b

                        index_i_c = int(i_s/k_x - actual_offset_x_c[0][indices_y_c[0][int(j_s/k_y)]])
                        index_y_c = indices_y_c[0][int(j_s/k_y)]

                        index_i_m = int(i_s/k_x - actual_offset_x_m[0][indices_y_m[0][int(j_s/k_y)]])
                        index_y_m = indices_y_m[0][int(j_s/k_y)]

                        index_i_y = int(i_s/k_x - actual_offset_x_y[0][indices_y_y[0][int(j_s/k_y)]])
                        index_y_y = indices_y_y[0][int(j_s/k_y)]

                        index_i_k = int(i_s/k_x - actual_offset_x_k[0][indices_y_k[0][int(j_s/k_y)]])
                        index_y_k = indices_y_k[0][int(j_s/k_y)]


                        if i_s >= 0 and j_s>=0:
                            if (img_c[index_i_c+border_width][index_y_c+border_width] != 0):
                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))] = k_x * index_i_c + k_x * actual_offset_x_c[0][index_y_c] - i

                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2)) + 1] = k_y * actual_position_y_c[0][index_y_c]-j

                                data_cpt[m * yfan + n][
                                    3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                        block_size_co / 2)) + 2] = actual_D_c[0][index_y_c]

                            if (img_m[index_i_m+border_width][index_y_m+border_width] != 0):
                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2)) + xiaochicun * 3] = k_x * index_i_m + k_x * actual_offset_x_m[0][index_y_m] - i

                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2)) + xiaochicun * 3 + 1] = k_y * actual_position_y_m[0][index_y_m]-j

                                data_cpt[m * yfan + n][
                                    3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                        block_size_co / 2))+ xiaochicun * 3 + 2] = actual_D_m[0][index_y_m]

                            if (img_y[index_i_y+border_width][index_y_y+border_width] != 0):
                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))+ xiaochicun * 6] = k_x * index_i_y + k_x * actual_offset_x_y[0][index_y_y] - i

                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))+ xiaochicun * 6 + 1] = k_y * actual_position_y_y[0][index_y_y]-j

                                data_cpt[m * yfan + n][
                                    3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                        block_size_co / 2))+ xiaochicun * 6 + 2] = actual_D_y[0][index_y_y]

                            if (img_k[index_i_k+border_width][index_y_k+border_width] != 0):
                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))+ xiaochicun * 9] = k_x * index_i_k + k_x * actual_offset_x_k[0][index_y_k] - i

                                data_cpt[m * yfan + n][3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2)) + xiaochicun * 9 + 1] = k_y * actual_position_y_k[0][index_y_k]-j

                                data_cpt[m * yfan + n][
                                    3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                        block_size_co / 2)) + xiaochicun * 9 + 2] = actual_D_k[0][index_y_k]


        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

def create_data_1016(img,img2,xfan,yfan,k_x,k_y,block_size_ro,block_size_co,pro,pco,save_path,path_pentou,mode = 0):
    #用于处理更大尺寸的数据集  实验用

    if mode ==0:
        real_height = img2.height
        real_width = img2.width

        image_width = img.width
        image_height = img.height


        img = np.array(img)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k

        img2 = np.array(img2)
        r, g, b = img2[..., 0], img2[..., 1], img2[..., 2]
        img_r = r
        img_g = g
        img_b = b

        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        # 实际位置
        actual_position_y_c = np.arange(0, actual_offset_y_c.shape[1]) + actual_offset_y_c
        # 利用 argsort 来获取排序后的索引
        indices_y_c = np.argsort(actual_position_y_c)

        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8

        # 实际位置
        actual_position_y_m = np.arange(0, actual_offset_y_m.shape[1]) + actual_offset_y_m
        # 利用 argsort 来获取排序后的索引
        indices_y_m = np.argsort(actual_position_y_m)

        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        # 实际位置
        actual_position_y_y = np.arange(0, actual_offset_y_y.shape[1]) + actual_offset_y_y
        # 利用 argsort 来获取排序后的索引
        indices_y_y = np.argsort(actual_position_y_y)

        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        # 实际位置
        actual_position_y_k = np.arange(0, actual_offset_y_k.shape[1]) + actual_offset_y_k
        # 利用 argsort 来获取排序后的索引
        indices_y_k = np.argsort(actual_position_y_k)





        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8


        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8


        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8


        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8



        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)

        actual_D_c = np.transpose(actual_D_c) / 16

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)

        actual_D_m = np.transpose(actual_D_m) / 16

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)

        actual_D_y = np.transpose(actual_D_y) / 16

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)

        actual_D_k = np.transpose(actual_D_k) / 16

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12 + 3

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))
        # 生成随机数组
        random_x = np.random.randint(1, real_height - pro, size=(1, xfan * yfan))
        random_y = np.random.randint(1, real_width - pco, size=(1, xfan * yfan))

        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                index_date = m * yfan + n
                i = random_x[0][index_date]
                j = random_y[0][index_date]

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):
                        i_s = i + a
                        j_s = j + b

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                            block_size_co / 2))


                        index_y_c = indices_y_c[0][int(j_s/k_y)]
                        index_i_c = int(i_s/k_x - actual_offset_x_c[0][index_y_c])

                        index_y_m = indices_y_m[0][int(j_s/k_y)]
                        index_i_m = int(i_s/k_x - actual_offset_x_m[0][index_y_m])

                        index_y_y = indices_y_y[0][int(j_s/k_y)]
                        index_i_y = int(i_s/k_x - actual_offset_x_y[0][index_y_y])

                        index_y_k = indices_y_k[0][int(j_s/k_y)]
                        index_i_k = int(i_s/k_x - actual_offset_x_k[0][index_y_k])



                        if (
                                index_y_c >= 0 and index_i_c >= 0 and index_y_c < image_width and index_i_c < image_height and
                                img_c[index_i_c][index_y_c] != 0):
                            data_cpt[index_date][index_date_y] = k_x * index_i_c + k_x * actual_offset_x_c[0][
                                index_y_c] - i

                            data_cpt[index_date][index_date_y + 1] = k_y * actual_offset_y_c[0][
                                index_y_c] + k_y * index_y_c - j

                            data_cpt[index_date][index_date_y + 2] = actual_D_c[0][index_y_c]

                        if (
                                index_y_m >= 0 and index_i_m >= 0 and index_y_m < image_width and index_i_m < image_height and
                                img_m[index_i_m][index_y_m] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 3] = k_x * index_i_m + k_x * \
                                                                                  actual_offset_x_m[0][index_y_m] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 1] = k_y * actual_offset_y_m[0][
                                index_y_m] + k_y * index_y_m - j

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][index_y_m]

                        if (
                                index_y_y >= 0 and index_i_y >= 0 and index_y_y < image_width and index_i_y < image_height and
                                img_y[index_i_y][index_y_y] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 6] = k_x * index_i_y + k_x * \
                                                                                  actual_offset_x_y[0][index_y_y] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 1] = k_y * actual_offset_y_y[0][
                                index_y_y] + k_y * index_y_y - j

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][index_y_y]

                        if (
                                index_y_k >= 0 and index_i_k >= 0 and index_y_k < image_width and index_i_k < image_height and
                                img_k[index_i_k][index_y_k] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 9] = k_x * index_i_k + k_x * \
                                                                                  actual_offset_x_k[0][index_y_k] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 1] = k_y * actual_offset_y_k[0][
                                index_y_k] + k_y * index_y_k - j

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][index_y_k]

                data_cpt[index_date][xiaochicun * 12] = img_r[i + pro][j + pco]
                data_cpt[index_date][xiaochicun * 12 + 1] = img_g[i + pro][j + pco]
                data_cpt[index_date][xiaochicun * 12 + 2] = img_b[i + pro][j + pco]

        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

    if mode ==1:
        image_width = img.width
        image_height = img.height

        xfan = int(img.height*k_x)
        yfan = int(img.width*k_y)


        img = np.array(img)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k

        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        # 实际位置
        actual_position_y_c = np.arange(0, actual_offset_y_c.shape[1]) + actual_offset_y_c
        # 利用 argsort 来获取排序后的索引
        indices_y_c = np.argsort(actual_position_y_c)

        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8

        # 实际位置
        actual_position_y_m = np.arange(0, actual_offset_y_m.shape[1]) + actual_offset_y_m
        # 利用 argsort 来获取排序后的索引
        indices_y_m = np.argsort(actual_position_y_m)

        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        # 实际位置
        actual_position_y_y = np.arange(0, actual_offset_y_y.shape[1]) + actual_offset_y_y
        # 利用 argsort 来获取排序后的索引
        indices_y_y = np.argsort(actual_position_y_y)

        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        # 实际位置
        actual_position_y_k = np.arange(0, actual_offset_y_k.shape[1]) + actual_offset_y_k
        # 利用 argsort 来获取排序后的索引
        indices_y_k = np.argsort(actual_position_y_k)

        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8


        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8


        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8


        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8



        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)

        actual_D_c = np.transpose(actual_D_c) / 16

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)

        actual_D_m = np.transpose(actual_D_m) / 16

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)

        actual_D_y = np.transpose(actual_D_y) / 16

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)

        actual_D_k = np.transpose(actual_D_k) / 16

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))

        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                index_date = m * yfan + n
                i = m
                j = n

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):
                        i_s = i + a
                        j_s =j + b

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))

                        index_y_c = indices_y_c[0][int(j_s/k_y)]
                        index_i_c = int(i_s/k_x - actual_offset_x_c[0][index_y_c])

                        index_y_m = indices_y_m[0][int(j_s/k_y)]
                        index_i_m = int(i_s/k_x - actual_offset_x_m[0][index_y_m])

                        index_y_y = indices_y_y[0][int(j_s/k_y)]
                        index_i_y = int(i_s/k_x - actual_offset_x_y[0][index_y_y])

                        index_y_k = indices_y_k[0][int(j_s/k_y)]
                        index_i_k = int(i_s/k_x - actual_offset_x_k[0][index_y_k])

                        if (index_y_c >= 0 and index_i_c >= 0 and index_y_c < image_width and index_i_c < image_height and img_c[index_i_c][index_y_c] != 0):
                            data_cpt[index_date][index_date_y] = k_x * index_i_c + k_x * actual_offset_x_c[0][
                                index_y_c] - i

                            data_cpt[index_date][index_date_y + 1] = k_y * actual_offset_y_c[0][
                                index_y_c] + k_y * index_y_c - j

                            data_cpt[index_date][index_date_y + 2] = actual_D_c[0][index_y_c]

                        if (index_y_m >= 0 and index_i_m >= 0 and  index_y_m < image_width and index_i_m < image_height and img_m[index_i_m][index_y_m] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 3] = k_x * index_i_m + k_x * \
                                                                                  actual_offset_x_m[0][index_y_m] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 1] = k_y * actual_offset_y_m[0][
                                index_y_m] + k_y * index_y_m - j

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][index_y_m]

                        if (index_y_y >= 0 and index_i_y >= 0 and  index_y_y < image_width and index_i_y < image_height and img_y[index_i_y][index_y_y] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 6] = k_x * index_i_y + k_x * \
                                                                                  actual_offset_x_y[0][index_y_y] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 1] = k_y * actual_offset_y_y[0][
                                index_y_y] + k_y * index_y_y - j

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][index_y_y]

                        if (index_y_k >= 0 and index_i_k >= 0 and index_y_k < image_width and index_i_k < image_height and img_k[index_i_k][index_y_k] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 9] = k_x * index_i_k + k_x * \
                                                                                  actual_offset_x_k[0][index_y_k] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 1] = k_y * actual_offset_y_k[0][
                                index_y_k] + k_y * index_y_k - j

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][index_y_k]






        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

def create_data_1017(img,img2,xfan,yfan,k_x,k_y,block_size_ro,block_size_co,pro,pco,save_path,path_pentou,mode = 0):
    #用于处理更大尺寸的数据集  实验用

    if mode ==0:
        real_height = img2.height
        real_width = img2.width

        image_width = img.width
        image_height = img.height


        img = np.array(img)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k

        img2 = np.array(img2)
        r, g, b = img2[..., 0], img2[..., 1], img2[..., 2]
        img_r = r
        img_g = g
        img_b = b

        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8




        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8


        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8




        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8





        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8


        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8


        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8


        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8



        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)

        actual_D_c = np.transpose(actual_D_c) / 16

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)

        actual_D_m = np.transpose(actual_D_m) / 16

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)

        actual_D_y = np.transpose(actual_D_y) / 16

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)

        actual_D_k = np.transpose(actual_D_k) / 16

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12 + 3

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))
        # 生成随机数组
        random_x = np.random.randint(1, real_height - pro, size=(1, xfan * yfan))
        random_y = np.random.randint(1, real_width - pco, size=(1, xfan * yfan))

        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                index_date = m * yfan + n
                i = random_x[0][index_date]
                j = random_y[0][index_date]

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):
                        i_s = int(i/k_x) + a
                        j_s = int(j/k_y) + b

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                            block_size_co / 2))

                        index_y_c = j_s + (2) * int(j_s / 2048)
                        index_i_c = i_s

                        index_y_m = j_s + (2) * int(j_s / 2048)
                        index_i_m = i_s

                        index_y_y = j_s + (2) * int(j_s / 2048)
                        index_i_y = i_s

                        index_y_k = j_s + (3) * int(j_s / 2048)
                        index_i_k = i_s
                        if (
                                index_y_c >= 0 and index_i_c >= 0 and index_y_c < image_width and index_i_c < image_height and
                                img_c[index_i_c][index_y_c] != 0):
                            data_cpt[index_date][index_date_y] = k_x * index_i_c + k_x * actual_offset_x_c[0][
                                index_y_c] - i

                            data_cpt[index_date][index_date_y + 1] = k_y * actual_offset_y_c[0][
                                index_y_c] + k_y * index_y_c - j

                            data_cpt[index_date][index_date_y + 2] = actual_D_c[0][index_y_c]

                        if (
                                index_y_m >= 0 and index_i_m >= 0 and index_y_m < image_width and index_i_m < image_height and
                                img_m[index_i_m][index_y_m] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 3] = k_x * index_i_m + k_x * \
                                                                                  actual_offset_x_m[0][index_y_m] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 1] = k_y * actual_offset_y_m[0][
                                index_y_m] + k_y * index_y_m - j

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][index_y_m]

                        if (
                                index_y_y >= 0 and index_i_y >= 0 and index_y_y < image_width and index_i_y < image_height and
                                img_y[index_i_y][index_y_y] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 6] = k_x * index_i_y + k_x * \
                                                                                  actual_offset_x_y[0][index_y_y] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 1] = k_y * actual_offset_y_y[0][
                                index_y_y] + k_y * index_y_y - j

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][index_y_y]

                        if (
                                index_y_k >= 0 and index_i_k >= 0 and index_y_k < image_width and index_i_k < image_height and
                                img_k[index_i_k][index_y_k] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 9] = k_x * index_i_k + k_x * \
                                                                                  actual_offset_x_k[0][index_y_k] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 1] = k_y * actual_offset_y_k[0][
                                index_y_k] + k_y * index_y_k - j

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][index_y_k]

                data_cpt[index_date][xiaochicun * 12] = img_r[i + pro][j + pco]
                data_cpt[index_date][xiaochicun * 12 + 1] = img_g[i + pro][j + pco]
                data_cpt[index_date][xiaochicun * 12 + 2] = img_b[i + pro][j + pco]

        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

    if mode ==1:
        image_width = img.width
        image_height = img.height

        xfan = int(img.height*k_x)
        yfan = int(img.width*k_y)


        img = np.array(img)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k

        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8


        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8

        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8

        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8

        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8


        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8


        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8


        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8



        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)

        actual_D_c = np.transpose(actual_D_c) / 16

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)

        actual_D_m = np.transpose(actual_D_m) / 16

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)

        actual_D_y = np.transpose(actual_D_y) / 16

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)

        actual_D_k = np.transpose(actual_D_k) / 16

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))

        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                index_date = m * yfan + n
                i = m
                j = n

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):
                        i_s = i + a
                        j_s =j + b

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                                    block_size_co / 2))

                        index_y_c = j_s+(2)*int(j_s/2048)
                        index_i_c = int(i_s/k_x)

                        index_y_m = j_s+(2)*int(j_s/2048)
                        index_i_m = int(i_s/k_x)

                        index_y_y = j_s+(2)*int(j_s/2048)
                        index_i_y = int(i_s/k_x)

                        index_y_k = j_s+(3)*int(j_s/2048)
                        index_i_k = int(i_s/k_x)

                        if (index_y_c >= 0 and index_i_c >= 0 and index_y_c < image_width and index_i_c < image_height and img_c[index_i_c][index_y_c] != 0):
                            data_cpt[index_date][index_date_y] = k_x * index_i_c + k_x * actual_offset_x_c[0][
                                index_y_c] - i

                            data_cpt[index_date][index_date_y + 1] = k_y * actual_offset_y_c[0][
                                index_y_c] + k_y * index_y_c - j

                            data_cpt[index_date][index_date_y + 2] = actual_D_c[0][index_y_c]

                        if (index_y_m >= 0 and index_i_m >= 0 and  index_y_m < image_width and index_i_m < image_height and img_m[index_i_m][index_y_m] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 3] = k_x * index_i_m + k_x * \
                                                                                  actual_offset_x_m[0][index_y_m] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 1] = k_y * actual_offset_y_m[0][
                                index_y_m] + k_y * index_y_m - j

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][index_y_m]

                        if (index_y_y >= 0 and index_i_y >= 0 and  index_y_y < image_width and index_i_y < image_height and img_y[index_i_y][index_y_y] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 6] = k_x * index_i_y + k_x * \
                                                                                  actual_offset_x_y[0][index_y_y] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 1] = k_y * actual_offset_y_y[0][
                                index_y_y] + k_y * index_y_y - j

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][index_y_y]

                        if (index_y_k >= 0 and index_i_k >= 0 and index_y_k < image_width and index_i_k < image_height and img_k[index_i_k][index_y_k] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 9] = k_x * index_i_k + k_x * \
                                                                                  actual_offset_x_k[0][index_y_k] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 1] = k_y * actual_offset_y_k[0][
                                index_y_k] + k_y * index_y_k - j

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][index_y_k]






        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

def create_data_1019(img,img2,xfan,yfan,k_x,k_y,block_size_ro,block_size_co,pro,pco,save_path,path_pentou,mode = 0):
    #用于处理更大尺寸的数据集  实验用
    #0123修改int为round
    """
    Description of each parameter:

    img: numpy array
        The halftone CMYK image that is processed. This image is composed of cyan, magenta, yellow, and black channels and is typically used in the printing process.

    img2: numpy array
        The scanned RGB image, corresponding to `img`. This image represents the actual printed result captured by a scanner, typically containing red, green, and blue color channels.

    xfan: int
        The number of sample points in the x-direction (width) for generating the dataset. This defines how many points will be randomly selected across the entire CMYK image when creating a training set.

    yfan: int
        The number of sample points in the y-direction (height). Similar to `xfan`, it specifies the number of points to sample vertically.

    k_x: float
        The stretch ratio of the paper in the x-direction. This accounts for distortions during printing, as the paper may stretch horizontally during the printing process.

    k_y: float
        The stretch ratio of the paper in the y-direction. This adjusts for vertical stretching of the paper during printing.

    block_size_ro: int
        The block size in the row dimension (height). This represents how many pixels in the CMYK halftone image influence the grayscale value of a single point in the continuous tone image.

    block_size_co: int
        The block size in the column dimension (width). This is the number of pixels in the CMYK halftone image influencing the grayscale value at a given point, similar to `block_size_ro`.

    pro: int
        The crop margin in the row dimension (height). This parameter helps compensate for differences in size between the halftone CMYK image and the scanned RGB image, allowing for necessary trimming when generating the training set.

    pco: int
        The crop margin in the column dimension (width). This is used to trim excess areas in the RGB image that do not perfectly align with the CMYK image due to scanning inaccuracies or paper stretching.

    save_path: str
        The file path where the processed dataset will be saved after generation.

    path_pentou: str
        The file path to the printhead data, which includes information such as nozzle positions and configurations that are critical for accurately mapping the printed ink dot distribution.

    mode: int, optional, default=0
        This parameter determines the operational mode. When set to 0 (default), the function operates in training mode, using the scanned RGB image (`img2`) for generating the dataset. When set to 1, the function runs in dataset mode, where the scanned image is not used, and the focus is purely on the halftone CMYK data.
    """

    if mode ==0:

        image_width = img.width
        image_height = img.height

        # 加一圈白边
        border_width = 2
        img = kuobaibian(img, border_width)

        img = np.array(img)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k

        img2 = np.array(img2)
        r, g, b = img2[..., 0], img2[..., 1], img2[..., 2]
        img_r = r
        img_g = g
        img_b = b



        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        actual_offset_y_c = yzeng(actual_offset_y_c, border_width)



        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8
        actual_offset_y_m = yzeng(actual_offset_y_m, border_width)


        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        actual_offset_y_y = yzeng(actual_offset_y_y, border_width)



        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        actual_offset_y_k = yzeng(actual_offset_y_k, border_width)




        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8
        actual_offset_x_c = yzeng(actual_offset_x_c, border_width)


        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8
        actual_offset_x_m = yzeng(actual_offset_x_m, border_width)

        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8
        actual_offset_x_y = yzeng(actual_offset_x_y, border_width)

        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8
        actual_offset_x_k = yzeng(actual_offset_x_k, border_width)


        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)
        actual_D_c = np.transpose(actual_D_c) / 16
        actual_D_c = yzeng(actual_D_c, border_width)

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)
        actual_D_m = np.transpose(actual_D_m) / 16
        actual_D_m = yzeng(actual_D_m, border_width)

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)
        actual_D_y = np.transpose(actual_D_y) / 16
        actual_D_y = yzeng(actual_D_y, border_width)

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)
        actual_D_k = np.transpose(actual_D_k) / 16
        actual_D_k = yzeng(actual_D_k, border_width)

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12 + 3

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))
        # 生成随机数组
        random_x = np.random.randint(1, int(image_height*k_x) - pro, size=(1, xfan * yfan))
        random_y = np.random.randint(1, int(image_width*k_y) - pco, size=(1, xfan * yfan))

        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                index_date = m * yfan + n
                i = random_x[0][index_date]
                j = random_y[0][index_date]
                j_c = round(j / k_y)
                j_m = j_c
                j_y = j_c
                j_k = j_c
                i_s = round(i / k_x)

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):
                        i_s = i_s + a

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                            block_size_co / 2))

                        index_y_c = j_c+b
                        index_i_c = i_s

                        index_y_m = j_m+b
                        index_i_m = i_s

                        index_y_y = j_y+b
                        index_i_y = i_s

                        index_y_k = j_k+b
                        index_i_k = i_s

                        if (img_c[index_i_c+border_width][index_y_c+border_width] != 0):
                            data_cpt[index_date][index_date_y] = k_x * index_i_c + k_x * actual_offset_x_c[0][
                                index_y_c+border_width] - i

                            data_cpt[index_date][index_date_y + 1] = k_y * actual_offset_y_c[0][
                                index_y_c+border_width] + k_y * index_y_c - j

                            data_cpt[index_date][index_date_y + 2] = actual_D_c[0][index_y_c+border_width]

                        if (img_m[index_i_m+border_width][index_y_m+border_width] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 3] = k_x * index_i_m + k_x * \
                                                                                  actual_offset_x_m[0][index_y_m+border_width] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 1] = k_y * actual_offset_y_m[0][
                                index_y_m+border_width] + k_y * index_y_m - j

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][index_y_m+border_width]

                        if (img_y[index_i_y+border_width][index_y_y+border_width] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 6] = k_x * index_i_y + k_x * \
                                                                                  actual_offset_x_y[0][index_y_y+border_width] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 1] = k_y * actual_offset_y_y[0][
                                index_y_y+border_width] + k_y * index_y_y - j

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][index_y_y+border_width]

                        if (img_k[index_i_k+border_width][index_y_k+border_width] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 9] = k_x * index_i_k + k_x * \
                                                                                  actual_offset_x_k[0][index_y_k+border_width] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 1] = k_y * actual_offset_y_k[0][
                                index_y_k+border_width] + k_y * index_y_k - j

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][index_y_k+border_width]

                data_cpt[index_date][xiaochicun * 12] = img_r[i + pro][j + pco]
                data_cpt[index_date][xiaochicun * 12 + 1] = img_g[i + pro][j + pco]
                data_cpt[index_date][xiaochicun * 12 + 2] = img_b[i + pro][j + pco]

        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

    if mode ==1:
        image_width = img.width
        image_height = img.height

        xfan = int(img.height*k_x)
        yfan = int(img.width*k_y)

        # 加一圈白边
        border_width = 2
        img = kuobaibian(img, border_width)

        img = np.array(img)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k

        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        actual_offset_y_c = yzeng(actual_offset_y_c, border_width)

        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8
        actual_offset_y_m = yzeng(actual_offset_y_m, border_width)

        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        actual_offset_y_y = yzeng(actual_offset_y_y, border_width)

        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        actual_offset_y_k = yzeng(actual_offset_y_k, border_width)

        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8
        actual_offset_x_c = yzeng(actual_offset_x_c, border_width)

        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8
        actual_offset_x_m = yzeng(actual_offset_x_m, border_width)

        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8
        actual_offset_x_y = yzeng(actual_offset_x_y, border_width)

        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8
        actual_offset_x_k = yzeng(actual_offset_x_k, border_width)

        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)
        actual_D_c = np.transpose(actual_D_c) / 16
        actual_D_c = yzeng(actual_D_c, border_width)

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)
        actual_D_m = np.transpose(actual_D_m) / 16
        actual_D_m = yzeng(actual_D_m, border_width)

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)
        actual_D_y = np.transpose(actual_D_y) / 16
        actual_D_y = yzeng(actual_D_y, border_width)

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)
        actual_D_k = np.transpose(actual_D_k) / 16
        actual_D_k = yzeng(actual_D_k, border_width)


        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))

        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                index_date = m * yfan + n
                i = m
                j = n
                j_c = round(j / k_y)
                j_m = j_c
                j_y = j_c
                j_k = j_c
                i_s = round(i/k_x)
                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):
                        i_s = i_s + a

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                            block_size_co / 2))

                        index_y_c = j_c+b
                        index_i_c = i_s

                        index_y_m = j_m+b
                        index_i_m = i_s

                        index_y_y = j_y+b
                        index_i_y = i_s

                        index_y_k = j_k+b
                        index_i_k = i_s

                        if (img_c[index_i_c+border_width][index_y_c+border_width] != 0):
                            data_cpt[index_date][index_date_y] = k_x * index_i_c + k_x * actual_offset_x_c[0][
                                index_y_c+border_width] - i

                            data_cpt[index_date][index_date_y + 1] = k_y * actual_offset_y_c[0][
                                index_y_c+border_width] + k_y * index_y_c - j

                            data_cpt[index_date][index_date_y + 2] = actual_D_c[0][index_y_c+border_width]

                        if (img_m[index_i_m+border_width][index_y_m+border_width] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 3] = k_x * index_i_m + k_x * \
                                                                                  actual_offset_x_m[0][index_y_m+border_width] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 1] = k_y * actual_offset_y_m[0][
                                index_y_m+border_width] + k_y * index_y_m - j

                            data_cpt[index_date][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][index_y_m+border_width]

                        if (img_y[index_i_y+border_width][index_y_y+border_width] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 6] = k_x * index_i_y + k_x * \
                                                                                  actual_offset_x_y[0][index_y_y+border_width] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 1] = k_y * actual_offset_y_y[0][
                                index_y_y+border_width] + k_y * index_y_y - j

                            data_cpt[index_date][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][index_y_y+border_width]

                        if (img_k[index_i_k+border_width][index_y_k+border_width] != 0):
                            data_cpt[index_date][index_date_y + xiaochicun * 9] = k_x * index_i_k + k_x * \
                                                                                  actual_offset_x_k[0][index_y_k+border_width] - i

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 1] = k_y * actual_offset_y_k[0][
                                index_y_k+border_width] + k_y * index_y_k - j

                            data_cpt[index_date][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][index_y_k+border_width]








        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

def create_data_size_shiyan_suijisenlin(img,img2,xfan,yfan,k_x,k_y,block_size_ro,block_size_co,pro,pco,save_path,path_pentou,mode = 0):
    if mode == 0:
        real_height = img.height
        real_width = img.width

        # 加一圈白边
        border_width = 2
        img_with_border = kuobaibian(img, border_width)


        img = np.array(img_with_border)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k

        img2 = np.array(img2)
        r, g, b = img2[..., 0], img2[..., 1], img2[..., 2]
        img_r = r
        img_g = g
        img_b = b

        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        actual_offset_y_c = yzeng(actual_offset_y_c, border_width)

        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8
        actual_offset_y_m = yzeng(actual_offset_y_m, border_width)

        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        actual_offset_y_y = yzeng(actual_offset_y_y, border_width)

        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        actual_offset_y_k = yzeng(actual_offset_y_k, border_width)

        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8
        actual_offset_x_c = yzeng(actual_offset_x_c, border_width)

        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8
        actual_offset_x_m = yzeng(actual_offset_x_m, border_width)

        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8
        actual_offset_x_y = yzeng(actual_offset_x_y, border_width)

        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8
        actual_offset_x_k = yzeng(actual_offset_x_k, border_width)

        # 创建实际大小D(半径)
        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)
        actual_D_c = np.transpose(actual_D_c) / 16
        actual_D_c = yzeng(actual_D_c, border_width)

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)
        actual_D_m = np.transpose(actual_D_m) / 16
        actual_D_m = yzeng(actual_D_m, border_width)

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)
        actual_D_y = np.transpose(actual_D_y) / 16
        actual_D_y = yzeng(actual_D_y, border_width)

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)
        actual_D_k = np.transpose(actual_D_k) / 16
        actual_D_k = yzeng(actual_D_k, border_width)

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        dmax = int(dx_c + dx_m + dx_y + dx_k)

        xiaochicun = block_size_co * block_size_ro
        chicun = xiaochicun * 12 + 3

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun))
        # 生成随机数组
        random_x = np.random.randint(1, real_height - pro, size=(1, xfan * yfan))
        random_y = np.random.randint(1, real_width - pco, size=(1, xfan * yfan))

        for m in range(dmax, xfan + dmax):
            for n in range(0, yfan):
                i = random_x[0][m * yfan + n]
                j = random_y[0][m * yfan + n]

                i_c = int(i / k_x)
                i_m = int(i / k_x)
                i_y = int(i / k_x)
                i_k = int(i / k_x)

                i_rc = i_c + dx_c
                j_r = int(j / k_y)

                index_data_x = m * yfan + n

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                            block_size_co / 2))
                        # c

                        if (img_c[i_c + a + border_width][j_r + b + border_width] != 0):
                            data_cpt[index_data_x][index_date_y] = k_x * (i_rc + a) + actual_offset_x_c[0][
                                j_r + border_width + b] - i
                            data_cpt[index_data_x][index_date_y + 1] = k_y * (j_r + b) + \
                                                                       actual_offset_y_c[0][
                                                                           j_r + border_width + b] - j
                            data_cpt[index_data_x][index_date_y + 2] = \
                                actual_D_c[0][
                                    j_r + border_width + b]

                        # m

                        if (img_m[i_m + a + border_width][j_r + b + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 3] = k_x * (i_rc + a) + \
                                                                                    actual_offset_x_m[0][
                                                                                        j_r + border_width + b] - i
                            data_cpt[index_data_x][index_date_y + xiaochicun * 3 + 1] = k_y * (j_r + b) + \
                                                                                        actual_offset_y_m[0][
                                                                                            j_r + border_width + b] - j
                            data_cpt[index_data_x][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][
                                j_r + border_width + b]

                        # y

                        if (img_y[i_y + a + border_width][j_r + b + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 6] = k_x * (i_rc + a) + \
                                                                                    actual_offset_x_y[0][
                                                                                        j_r + border_width + b] - i
                            data_cpt[index_data_x][index_date_y + xiaochicun * 6 + 1] = k_y * (j_r + b) + \
                                                                                        actual_offset_y_y[0][
                                                                                            j_r + border_width + b] - j
                            data_cpt[index_data_x][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][
                                j_r + border_width + b]

                        # k

                        if (img_k[i_k + a + border_width][j_r + b + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9] = k_x * (i_rc + a) + \
                                                                                    actual_offset_x_k[0][
                                                                                        j_r + border_width + b] - i
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9 + 1] = k_y * (j_r + b) + \
                                                                                        actual_offset_y_k[0][
                                                                                            j_r + border_width + b] - j
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][
                                j_r + border_width + b]


                data_cpt[index_data_x][xiaochicun * 12] = img_r[i + pro][j + pco]
                data_cpt[index_data_x][xiaochicun * 12 + 1] = img_g[i + pro][j + pco]
                data_cpt[index_data_x][xiaochicun * 12 + 2] = img_b[i + pro][j + pco]

        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, mode='a', header=False, index=False)

    if mode == 1:

        img_width, img_height = img.size

        xfan = int(img_height*k_x)
        yfan = int(img_width*k_y)

        # 加一圈白边
        border_width = 2
        img_with_border = kuobaibian(img, border_width)

        # 读取长宽
        width, height = img_with_border.size

        img = np.array(img_with_border)
        c, m, y, k = img[..., 0], img[..., 1], img[..., 2], img[..., 3]
        img_c = c
        img_m = m
        img_y = y
        img_k = k



        pentou = pd.read_excel(path_pentou)

        # 创建实际偏移Δy行向量
        actual_offset_y_c = np.array(pentou.iloc[:, 6:7].values)
        actual_offset_y_c = np.transpose(actual_offset_y_c) / 8
        actual_offset_y_c = yzeng(actual_offset_y_c, border_width)

        actual_offset_y_m = np.array(pentou.iloc[:, 11:12].values)
        actual_offset_y_m = np.transpose(actual_offset_y_m) / 8
        actual_offset_y_m = yzeng(actual_offset_y_m, border_width)

        actual_offset_y_y = np.array(pentou.iloc[:, 16:17].values)
        actual_offset_y_y = np.transpose(actual_offset_y_y) / 8
        actual_offset_y_y = yzeng(actual_offset_y_y, border_width)

        actual_offset_y_k = np.array(pentou.iloc[:, 21:22].values)
        actual_offset_y_k = np.transpose(actual_offset_y_k) / 8
        actual_offset_y_k = yzeng(actual_offset_y_k, border_width)

        # 创建实际偏移Δx列向量
        actual_offset_x_c = np.array(pentou.iloc[:, 5:6].values)
        actual_offset_x_c = np.transpose(actual_offset_x_c) / 8
        actual_offset_x_c = yzeng(actual_offset_x_c, border_width)

        actual_offset_x_m = np.array(pentou.iloc[:, 10:11].values)
        actual_offset_x_m = np.transpose(actual_offset_x_m) / 8
        actual_offset_x_m = yzeng(actual_offset_x_m, border_width)

        actual_offset_x_y = np.array(pentou.iloc[:, 15:16].values)
        actual_offset_x_y = np.transpose(actual_offset_x_y) / 8
        actual_offset_x_y = yzeng(actual_offset_x_y, border_width)

        actual_offset_x_k = np.array(pentou.iloc[:, 20:21].values)
        actual_offset_x_k = np.transpose(actual_offset_x_k) / 8
        actual_offset_x_k = yzeng(actual_offset_x_k, border_width)


        # 创建实际大小D(半径)
        actual_D_c = np.array(pentou.iloc[:, 7:8].values)
        actual_D_c = np.transpose(actual_D_c) / 16
        actual_D_c = yzeng(actual_D_c, border_width)

        actual_D_m = np.array(pentou.iloc[:, 12:13].values)
        actual_D_m = np.transpose(actual_D_m) / 16
        actual_D_m = yzeng(actual_D_m, border_width)

        actual_D_y = np.array(pentou.iloc[:, 17:18].values)
        actual_D_y = np.transpose(actual_D_y) / 16
        actual_D_y = yzeng(actual_D_y, border_width)

        actual_D_k = np.array(pentou.iloc[:, 22:23].values)
        actual_D_k = np.transpose(actual_D_k) / 16
        actual_D_k = yzeng(actual_D_k, border_width)

        xiaochicun = block_size_ro * block_size_co
        chicun = xiaochicun * 12

        # 创建数据集输入
        data_cpt = np.zeros((xfan * yfan, chicun + 3))

        # 套色偏移
        dx_c = 0
        dx_m = 0
        dx_y = 0
        dx_k = 0

        for m in range(0, xfan):
            for n in range(0, yfan):
                i = m
                j = n

                i_c = int(i / k_x)
                i_m = int(i / k_x)
                i_y = int(i / k_x)
                i_k = int(i / k_x)

                i_rc = i_c + dx_c
                j_r = int(j / k_y)
                index_data_x = m * yfan + n

                # 随机森林用
                for a in range(- int(block_size_ro / 2), block_size_ro - int(block_size_ro / 2)):
                    for b in range(- int(block_size_co / 2), block_size_co - int(block_size_co / 2)):

                        index_date_y = 3 * (block_size_co * (a + int(block_size_ro / 2)) + b + int(
                            block_size_co / 2))
                        # c

                        if (img_c[i_c + a + border_width][j_r + b + border_width] != 0):
                            data_cpt[index_data_x][index_date_y] = k_x * (i_rc + a) + actual_offset_x_c[0][
                                j_r + border_width + b] - i
                            data_cpt[index_data_x][index_date_y + 1] = k_y * (j_r + b) + \
                                                                       actual_offset_y_c[0][
                                                                           j_r + border_width + b] - j
                            data_cpt[index_data_x][index_date_y + 2] = \
                                actual_D_c[0][
                                    j_r + border_width + b]

                        # m

                        if (img_m[i_m + a + border_width][j_r + b + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 3] = k_x * (i_rc + a) + \
                                                                                    actual_offset_x_m[0][
                                                                                        j_r + border_width + b] - i
                            data_cpt[index_data_x][index_date_y + xiaochicun * 3 + 1] = k_y * (j_r + b) + \
                                                                                        actual_offset_y_m[0][
                                                                                            j_r + border_width + b] - j
                            data_cpt[index_data_x][index_date_y + xiaochicun * 3 + 2] = actual_D_m[0][
                                j_r + border_width + b]

                        # y

                        if (img_y[i_y + a + border_width][j_r + b + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 6] = k_x * (i_rc + a) + \
                                                                                    actual_offset_x_y[0][
                                                                                        j_r + border_width + b] - i
                            data_cpt[index_data_x][index_date_y + xiaochicun * 6 + 1] = k_y * (j_r + b) + \
                                                                                        actual_offset_y_y[0][
                                                                                            j_r + border_width + b] - j
                            data_cpt[index_data_x][index_date_y + xiaochicun * 6 + 2] = actual_D_y[0][
                                j_r + border_width + b]

                        # k

                        if (img_k[i_k + a + border_width][j_r + b + border_width] != 0):
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9] = k_x * (i_rc + a) + \
                                                                                    actual_offset_x_k[0][
                                                                                        j_r + border_width + b] - i
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9 + 1] = k_y * (j_r + b) + \
                                                                                        actual_offset_y_k[0][
                                                                                            j_r + border_width + b] - j
                            data_cpt[index_data_x][index_date_y + xiaochicun * 9 + 2] = actual_D_k[0][
                                j_r + border_width + b]



        df = pd.DataFrame(data_cpt)
        df.to_csv(save_path, index=False)

def cut_image(img,width,height):
    # 保留图片的前2000列
    left = 0
    top = 0

    # 裁剪图片
    img = img.crop((left, top, left + width, top + height))

    return img

#The stretch ratio of the paper in the x and y directions
k_x =1.001825
k_y =1.004221

# k_x =1.00287187
# k_y =1.0036842

# k_x =1.00287187
# k_y =1.0036842

# Load the image,Change the path to your own directory. Please note that img is a CMYK halftone image, and img2 is the scanned image.

# img = Image.open("D:/zyw/shenjingwangluo/input/fruit2.tif").convert('CMYK')
# img = Image.open("D:/zyw/6.2/qiyu.tif").convert('CMYK')
#
# img2 = Image.open("D:/zyw/6.2/pipei/005d.tif")

img = Image.open("D:/zyw/6.2/sise2.tif").convert('CMYK')

img2 = Image.open("D:/zyw/6.2/pipei/006d1.tif")


# img = cut_image(img,4224,img.height)
# img2 = cut_image(img2,int(4224/k_y),int(img.height/k_x))

# This limits the sampling range.
# If it's for creating a training set, it will randomly
# select xfan * yfan points across the entire image.
# However, if it's for a test set, this parameter has no effect.
xfan = 6000
yfan = 6000

#The path to the printhead data
path_pentou = "D:/zyw/shenjingwangluo/input/副本一号喷头数据0603.xlsx"

# save_path = "D:/zyw/shenjingwangluo/output/ceshi_1024_fruit_2.csv"

#The path to save the dataset
# save_path = "E:/quanshujuji0118_2.csv"
save_path = "E:/fh_python/input/sjj1_new.csv"

# shujuji_create(img,img2,k_x,k_y,5,5,save_path,path_pentou)
# create_data_size_shiyan_suijisenlin(img,img2,xfan,yfan,k_x,k_y,5,5,0,0,save_path,path_pentou,mode = 0)

create_data_1019(img,img2,xfan,yfan,k_x,k_y,5,5,0,0,save_path,path_pentou,mode = 0)

# create_data_1017(img,img2,xfan,yfan,k_x,k_y,5,5,0,0,save_path,path_pentou,mode = 1)

# create_data_size_shiyan_suijisenlin(img,img2,xfan,yfan,k_x,k_y,5,5,0,0,save_path,path_pentou,1)

# shujuji_create_size_suijisenlin(img,img2,real_height,real_width,k_x,k_y,5,5,save_path,path_pentou)

# shujuji_yuce(img,real_height,real_width,1,1,5,5,save_path)

# create_data_size_shiyan(img,img2,xfan,yfan,0.5,1,1,7,7,0,0,save_path,path_pentou)


# shujuji_create_size(img,real_height,real_width,0.5,1,1,3,5,save_path,path_pentou)

# create_data_size(img,img2,xfan,yfan,0.5,1,1,7,save_path,path_pentou)

# create_data_single2(img,img2,xfan,yfan,0.5,real_height,real_width,save,path_pentou)
# shujuji_create(img,real_height,real_width,0.5,save_path,path_pentou)
# create_data_imp(img,img2,xfan,yfan,0.5,1,1,save,path_pentou)
# shujuji_create_imp(img,real_height,real_width,0.5,1,1,save_path,path_pentou)
