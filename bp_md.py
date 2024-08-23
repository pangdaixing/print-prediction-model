from tensorflow.keras.models import Model
from tensorflow.keras.layers import Dense, Input, Add, Activation
from tensorflow.keras import regularizers
from tensorflow.keras.models import Sequential
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import LearningRateScheduler
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error
import math
from sklearn.preprocessing import StandardScaler
import time


# 学习率退火函数
def lr_scheduler(epoch, lr):
    if epoch == 1000 or epoch == 2000:
        lr = lr * math.exp(-0.2)
    return lr


block_size_ro=5
block_size_co=5
# 读取 Excel 表格数据
# df = pd.read_csv("D:/zyw/shenjingwangluo/output/shiyan_1024_2.csv")
df = pd.read_csv('D:\zyw\shenjingwangluo\output/shiyan_final.csv')

# 提取前240列的数据
X_train = np.array(df.iloc[:, :block_size_ro*block_size_co*12].values)

# 提取后三列的数据
y_train = np.array(df.iloc[:, -3:].values)

# df = pd.read_csv("D:/zyw/shenjingwangluo/output/ceshi_1016_jiubei.csv")
#
# # 提取前240列的数据
#
# X_train_2 = np.array(df.iloc[:, :block_size_ro*block_size_co*12].values)
#
# y_train_2 = np.array(df.iloc[:, -3:].values)

# df = pd.read_csv("D:/zyw/shenjingwangluo/output/ceshi_1013_nvhai.csv")
#
# # 提取前240列的数据
#
# X_train_3 = np.array(df.iloc[:, :block_size_ro*block_size_co*12].values)
#
# y_train_3 = np.array(df.iloc[:, -3:].values)


# X_train = np.concatenate((X_train_1, X_train_2), axis=0)
#
# y_train = np.concatenate((y_train_1, y_train_2), axis=0)

# X_train = np.concatenate((X_train, X_train_3), axis=0)
#
# y_train = np.concatenate((y_train, y_train_3), axis=0)


X_test = X_train[0:500, :block_size_ro*block_size_co*12]


y_test = y_train[0:500, :]

# 输入层数和节点数
num_layers = 3
num_units = 200

# 创建输入层
inputs = Input(shape=(block_size_ro*block_size_co*12,))

x = Dense(num_units, activation='relu', kernel_regularizer=regularizers.l1(0.001) ,kernel_initializer='glorot_uniform')(inputs)  # 第一层隐藏层

# 添加额外的隐藏层
for _ in range(num_layers - 1):
    residual = x
    x = Dense(num_units, activation='relu', kernel_regularizer=regularizers.l1(0.001))(x)
    x = Add()([x, residual])  # 添加残差连接

outputs = Dense(3, activation='linear')(x)  # 输出层

# 创建模型
resnet = Model(inputs=inputs, outputs=outputs)
resnet.summary()

# 编译模型
resnet.compile(optimizer='Adam', loss='mean_squared_error')

# 设置学习率退火
lr_decay = LearningRateScheduler(lr_scheduler)

# 将数据转换为GPU可用的格式
X_train = np.array(X_train).astype('float32')
y_train = np.array(y_train).astype('float32')
X_val = np.array(X_test).astype('float32')
y_val = np.array(y_test).astype('float32')

# 记录开始时间
start_time = time.time()
# 训练模型
history = resnet.fit(X_train, y_train, epochs=3000, batch_size=40000, verbose=2, validation_data=(X_val, y_val), callbacks=[lr_decay])

# 记录结束时间
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time

# 打印结果和执行时间
print(f"Execution Time: {execution_time} seconds")

# 保存模型
resnet.save('D:/zyw/XUNLIAN/tranmodel/bp_best_model.h5')

# # 获取训练和验证过程中的损失值
# loss = history.history['loss']
# val_loss = history.history['val_loss']

# # 将损失值转换为对数刻度
# loss = np.log(loss)
# val_loss = np.log(val_loss)

# # 绘制训练和验证损失值随训练周期的变化曲线
# plt.plot(loss, label='Training Loss')
# plt.plot(val_loss, label='Validation Loss')
# plt.xlabel("Epochs")
# plt.ylabel("Loss")
# plt.title("Training and Validation Loss")
# plt.legend()
# plt.show()

# # 将训练和验证损失保存到Excel
# df_loss = pd.DataFrame({'Epoch': range(1, len(loss) + 1), 'Training Loss': loss, 'Validation Loss': val_loss})
# df_loss.to_excel('D:/zyw/XUNLIAN/tranmodel/training_validation_loss.xlsx', index=False)
#
# # 预测测试集
# X_test = X_train[:500, :]
# y_test = y_train[:500, :]
# y_pred = resnet.predict(X_test)
#
# # 绘制预测结果和真实值的散点图
# plt.scatter(y_test[:, 0], y_pred[:, 0])
# plt.xlabel("True values")
# plt.ylabel("Predicted values")
# plt.title("True vs Predicted Values")
# plt.show()
#
# # 计算均方误差
# mse = mean_squared_error(y_test, y_pred)
# print("均方误差:", mse)