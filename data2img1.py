from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Add, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
import xgboost as xgb
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
import tensorflow as tf

def mdn_loss(y_true, mdn_outputs):
    mu = mdn_outputs[:, :num_components]
    sigma = mdn_outputs[:, num_components:2 * num_components]
    coeff = mdn_outputs[:, 2 * num_components:]

    exponent = -0.5 * tf.reduce_sum(tf.square((y_true - mu) / sigma), axis=1)
    normalizer = tf.reduce_sum(coeff, axis=1)
    log_probs = exponent - tf.math.log(tf.maximum(normalizer, 1e-10))

    return -tf.reduce_mean(log_probs)


# # 加载保存的模型
# loaded_model = tf.keras.models.load_model('D:/zyw/XUNLIAN/tranmodel/trained_model_final.h5')
# loaded_model = tf.keras.models.load_model('D:/zyw/XUNLIAN/tranmodel/trained_model_final.h5')
# # 加载模型并注册自定义损失函数
# loaded_model = tf.keras.models.load_model('D:/zyw/XUNLIAN/tranmodel/mdn_model_final.h5', custom_objects={'mdn_loss': mdn_loss})
# 打印模型摘要
# loaded_model.summary()
# 加载模型
loaded_model = xgb.Booster()
loaded_model.load_model('D:/zyw/XUNLIAN/tranmodel/xgb_best_model.model')


df = pd.read_csv("D:/zyw/shenjingwangluo/output/ceshi_final.csv")


# 加载模型

k_x =1
k_y =1
# height = 2240
# width = 3069

height = 2240
width = 3024


range_fanwei = math.ceil(int(height*k_x)*int(k_y*width)/40000)

range_fanwei_te002=112

range_fanwei_te004=328

range_fanwei_nvhai=534

block_size_ro=5
block_size_co=5

for i in range(range_fanwei):

    X_train = np.array(df.iloc[i*40000:(i+1)*40000, :block_size_ro*block_size_co*12].values)

    # 预测结果
    y_pred = loaded_model.predict(xgb.DMatrix(X_train))

    # y_pred = loaded_model.predict(X_train)

    savepath='D:/zyw/XUNLIAN/output/y_pred_'+str(i)+'.txt'


    np.savetxt(savepath, y_pred, fmt='%d', delimiter=' ')


    print(i)


# X_train = np.array(df.iloc[534*40000:, :180].values)
# y_pred = loaded_model.predict(X_train)
#
#
# np.savetxt('F:/XUNLIAN/output/y_pred_111.txt', y_pred, fmt='%d', delimiter=' ')

