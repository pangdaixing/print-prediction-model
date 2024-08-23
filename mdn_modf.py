import tensorflow as tf
from tensorflow.keras.layers import Dense, Input, Activation, Lambda, Concatenate
from tensorflow.keras import regularizers, Model
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
import math
import time

block_size_ro = 5
block_size_co = 5

# 生成示例数据
num_samples = 40000
num_features = block_size_ro * block_size_co * 12
num_components = 3

# 读取 Excel 表格数据
df = pd.read_csv('D:/zyw/shenjingwangluo/output/shiyan_final.csv')

# 提取前240列的数据
X = np.array(df.iloc[0:num_samples, :num_features].values)

X_test = X[0:500, :num_features]

# 提取后三列的数据
y_true = np.array(df.iloc[0:num_samples, -3:].values)
y_test = y_true[0:500, :]

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y_true, test_size=0.2, random_state=42)

# 超参数设置
param_grid = {
    'num_layers': [1, 3, 5,7,9,11,13,15,17],
    'num_hidden_units': [300,500, 1000, 1500],
    'batch_size': [40000],
    'learning_rate': [0.01, 0.1, 0.5]
}

best_val_loss = float('inf')
best_params = None
results = []

for num_layers in param_grid['num_layers']:
    for num_hidden_units in param_grid['num_hidden_units']:
        for batch_size in param_grid['batch_size']:
            for learning_rate in param_grid['learning_rate']:
                print(f"Training with num_layers={num_layers}, num_hidden_units={num_hidden_units}, batch_size={batch_size}, learning_rate={learning_rate}")

                # 构建模型
                inputs = Input(shape=(num_features,))
                x = Dense(num_hidden_units, activation='relu')(inputs)
                for _ in range(num_layers - 1):
                    x = Dense(num_hidden_units, activation='relu')(x)

                # 分别输出均值、标准差和混合系数
                mu = Dense(num_components, activation='linear')(x)
                sigma = Dense(num_components, activation='softplus')(x)
                coeff = Dense(num_components, activation='softmax')(x)

                mdn_outputs = Concatenate()([mu, sigma, coeff])
                mdn = Model(inputs=inputs, outputs=mdn_outputs)

                # 自定义损失函数：负对数似然
                def mdn_loss(y_true, mdn_outputs):
                    mu = mdn_outputs[:, :num_components]
                    sigma = mdn_outputs[:, num_components:2 * num_components]
                    coeff = mdn_outputs[:, 2 * num_components:]

                    exponent = -0.5 * tf.reduce_sum(tf.square((y_true - mu) / sigma), axis=1)
                    normalizer = tf.reduce_sum(coeff, axis=1)
                    log_probs = exponent - tf.math.log(tf.maximum(normalizer, 1e-10))

                    return -tf.reduce_mean(log_probs)

                mdn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=mdn_loss)

                # 训练模型
                history = mdn.fit(X_train, y_train, epochs=50, batch_size=batch_size, verbose=2, validation_data=(X_val, y_val))

                # 获取验证损失（NLL）
                val_loss_nll = history.history['val_loss'][-1]
                results.append((num_layers, num_hidden_units, batch_size, learning_rate, val_loss_nll))

                if val_loss_nll < best_val_loss:
                    best_val_loss = val_loss_nll
                    best_params = (num_layers, num_hidden_units, batch_size, learning_rate)

print(f"Best params: {best_params} with validation loss (NLL): {best_val_loss}")

# 使用最佳参数训练最终模型
num_layers, num_hidden_units, batch_size, learning_rate = best_params
inputs = Input(shape=(num_features,))
x = Dense(num_hidden_units, activation='relu')(inputs)
for _ in range(num_layers - 1):
    x = Dense(num_hidden_units, activation='relu')(x)

mu = Dense(num_components, activation='linear')(x)
sigma = Dense(num_components, activation='softplus')(x)
coeff = Dense(num_components, activation='softmax')(x)

mdn_outputs = Concatenate()([mu, sigma, coeff])
mdn = Model(inputs=inputs, outputs=mdn_outputs)

mdn.compile(optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate), loss=mdn_loss)

# 记录开始时间
start_time = time.time()

history = mdn.fit(X_train, y_train, epochs=50000, batch_size=batch_size, verbose=2, validation_data=(X_val, y_val))

# 记录结束时间
end_time = time.time()

# 计算执行时间
execution_time = end_time - start_time
print(f"Execution Time: {execution_time} seconds")

# 获取训练和验证过程中的损失值
train_loss_nll = history.history['loss']
val_loss_nll = history.history['val_loss']

# 绘制训练和验证损失值随训练周期的变化曲线
plt.plot(train_loss_nll, label='Training Loss (NLL)')
plt.plot(val_loss_nll, label='Validation Loss (NLL)')
plt.xlabel("Epochs")
plt.ylabel("Loss (NLL)")
plt.title("Training and Validation Loss")
plt.legend()
plt.show()

# 将训练和验证损失保存到Excel
df_loss = pd.DataFrame({'Epoch': range(1, len(train_loss_nll) + 1), 'Training Loss (NLL)': train_loss_nll, 'Validation Loss (NLL)': val_loss_nll})
df_loss.to_excel('D:/zyw/XUNLIAN/tranmodel/mdn_training_validation_loss.xlsx', index=False)

# 保存超参数优化结果到Excel
df_results = pd.DataFrame(results, columns=['num_layers', 'num_hidden_units', 'batch_size', 'learning_rate', 'val_loss (NLL)'])
df_results.to_excel('D:/zyw/XUNLIAN/tranmodel/mdn_hyperparameter_optimization_results.xlsx', index=False)

# 预测结果
y_pred = mdn.predict(X_test)[:, :num_components]  # 只保留均值部分

# 绘制预测结果和真实值的散点图
plt.scatter(y_test[:, 0], y_pred[:, 0])
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("True vs Predicted Values")
plt.show()

# # 计算均方误差
# mse = mean_squared_error(y_test, y_pred)
# print("均方误差:", mse)

# 保存模型到文件
mdn.save('D:/zyw/XUNLIAN/tranmodel/mdn_model_1.h5')

# 可视化超参数优化结果
fig, ax = plt.subplots()
for learning_rate in param_grid['learning_rate']:
    subset = [r for r in results if r[3] == learning_rate]
    hidden_units = [r[1] for r in subset]
    val_losses = [r[4] for r in subset]
    ax.plot(hidden_units, val_losses, label=f'LR={learning_rate}')

ax.set_xlabel('Number of Hidden Units')
ax.set_ylabel('Validation Loss (NLL)')
ax.set_title('Validation Loss for Different Hyperparameter Combinations')
ax.legend()
plt.show()
