import xgboost as xgb
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error
import matplotlib.pyplot as plt
import time

block_size_ro = 5
block_size_co = 5
# 生成示例数据
num_samples = 40000

# 读取 Excel 表格数据
df = pd.read_csv('D:/zyw/shenjingwangluo/output/shiyan_final.csv')

# 提取前240列的数据
X = np.array(df.iloc[0:num_samples, :block_size_ro * block_size_co * 12].values)

# 提取后三列的数据
y = np.array(df.iloc[0:num_samples, -3:].values)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)

# 定义超参数网格
param_grid = {
    'learning_rate': [0.0001, 0.001, 0.01],
    'n_estimators': [500, 1000, 2000, 3000],
    'max_depth': [6, 12, 19, 22]
}

# 初始化XGBoost模型
xgb_model = xgb.XGBRegressor(objective='reg:squarederror')

# 使用GridSearchCV进行超参数优化
grid_search = GridSearchCV(estimator=xgb_model, param_grid=param_grid, cv=3, scoring='neg_mean_squared_error', verbose=2)
grid_search.fit(X_train, y_train)

# 输出最佳超参数组合
print(f"Best parameters found: {grid_search.best_params_}")
print(f"Best validation MSE: {-grid_search.best_score_}")

# 获取每次迭代的损失函数值
results = grid_search.cv_results_

# 将超参数优化结果保存到Excel
df_results = pd.DataFrame({
    'Param Learning Rate': results['param_learning_rate'],
    'Param N Estimators': results['param_n_estimators'],
    'Param Max Depth': results['param_max_depth'],
    'Mean Test MSE': -results['mean_test_score'],  # 取负数恢复MSE值
    'Std Test MSE': results['std_test_score'],
    'Rank Test Score': results['rank_test_score']
})
df_results.to_excel('D:/zyw/XUNLIAN/tranmodel/xgb_hyperparameter_optimization_results.xlsx', index=False)

# 使用最佳参数训练最终模型
best_model = grid_search.best_estimator_
best_model.fit(X_train, y_train, eval_set=[(X_train, y_train), (X_val, y_val)], eval_metric='rmse', verbose=2)

# 获取每次迭代的损失函数值（RMSE）
train_rmse = best_model.evals_result()['validation_0']['rmse']
val_rmse = best_model.evals_result()['validation_1']['rmse']

# 将RMSE转换为MSE
train_losses = np.square(train_rmse)
val_losses = np.square(val_rmse)

# 打印每次迭代的训练和验证损失函数值
for iteration, (train_loss, val_loss) in enumerate(zip(train_losses, val_losses), start=1):
    print(f"Iteration {iteration}: Train MSE = {train_loss:.4f}, Validation MSE = {val_loss:.4f}")

# 预测结果
y_pred = best_model.predict(X_val)

# 计算均方误差
mse = mean_squared_error(y_val, y_pred)
print("均方误差:", mse)

# 保存模型到文件
best_model.save_model('D:/zyw/XUNLIAN/tranmodel/xgb_best_model.model')

# 绘制预测结果和真实值的散点图
plt.scatter(y_val[:, 0], y_pred[:, 0])
plt.xlabel("True values")
plt.ylabel("Predicted values")
plt.title("Predicted vs. True Values")
plt.show()

# 将训练和验证损失保存到Excel
df_loss = pd.DataFrame({
    'Iteration': range(1, len(train_losses) + 1),
    'Train MSE': train_losses,
    'Validation MSE': val_losses
})
df_loss.to_excel('D:/zyw/XUNLIAN/tranmodel/xgb_training_validation_loss.xlsx', index=False)

# 将超参数优化结果保存到Excel并可视化
df_visualization = pd.DataFrame(results)
df_visualization.to_excel('D:/zyw/XUNLIAN/tranmodel/xgb_hyperparameter_optimization_visualization.xlsx', index=False)

# 可视化超参数优化结果
fig, ax = plt.subplots()
for learning_rate in param_grid['learning_rate']:
    subset = df_visualization[df_visualization['param_learning_rate'] == learning_rate]
    ax.plot(subset['param_n_estimators'], -subset['mean_test_score'], label=f'LR={learning_rate}')

ax.set_xlabel('Number of Estimators')
ax.set_ylabel('Mean Squared Error (MSE)')
ax.set_title('Validation MSE for Different Hyperparameter Combinations')
ax.legend()
plt.show()
