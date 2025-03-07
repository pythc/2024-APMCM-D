import pandas as pd
from statsmodels.tsa.ar_model import AutoReg
import numpy as np
import kaiwu as kw
import matplotlib.pyplot as plt
import time
import seaborn as sns

kw.license.init(user_id="72279506022334466", sdk_code="OGJy50Ww8yEoKmAA4EcFTB8bVX2ort")

np.random.seed(42)

data = {
    'demand': [9000, 9400, 9594, 9859, 9958, 10043, 10309, 10512, 10588]  # 月度需求数据
}

df = pd.DataFrame(data)

#AR(3)模型训练时间
start_time_ar = time.time()

#训练AR(3)模型
model = AutoReg(df['demand'], lags=3)
model_fitted = model.fit()

#AR(3)模型训练时间
end_time_ar = time.time()
ar_training_time = end_time_ar - start_time_ar
print(f"AR(3) 模型训练时间：{ar_training_time:.4f} 秒")

#输出AR(3)模型的回归系数
print("AR(3) 模型的回归系数：")
print(model_fitted.params)

#生成QUBO矩阵的目标函数改进版本
def generate_qubo_matrix_improved(coeffs, history_residual):
    n = len(coeffs)
    Q = np.zeros((n, n))


    history_residual = history_residual.reset_index(drop=True)

    #设计一个复合目标函数包括AR模型误差
    for i in range(n):
        for j in range(i, n):
            if i == j:
                Q[i][i] = coeffs[i] - history_residual[i]
            else:
                Q[i][j] = Q[j][i] = coeffs[i] * coeffs[j]

    return Q

#生成改进后的QUBO矩阵
coefficients = [model_fitted.params['const'], model_fitted.params['demand.L1'], model_fitted.params['demand.L2'],
                model_fitted.params['demand.L3']]
history_residual = model_fitted.resid  # 使用 AR(3) 模型的残差作为误差项
qubo_matrix_improved = generate_qubo_matrix_improved(coefficients, history_residual)


print("改进后的 QUBO 矩阵：")
print(qubo_matrix_improved)

#使用改进的QUBO矩阵进行优化
start_time_qubo = time.time()

solver = kw.classical.SimulatedAnnealingOptimizer(initial_temperature=500,
                                                  alpha=0.95,
                                                  cutoff_temperature=0.001,
                                                  iterations_per_t=500,
                                                  size_limit=10)
optimal_solution = solver.solve(qubo_matrix_improved)

#求解时间
end_time_qubo = time.time()
qubo_solving_time = end_time_qubo - start_time_qubo
print(f"QUBO 求解时间：{qubo_solving_time:.4f} 秒")

#输出最优解
print(f"Optimal Solution: {optimal_solution}")

#计算根据 QUBO 解预测的需求
if len(optimal_solution) > 0:
    optimal_solution_row = optimal_solution[0]
    predicted_demand_qubo = (optimal_solution_row[0] + optimal_solution_row[1] * df['demand'].iloc[-1] + \
                            optimal_solution_row[2] * df['demand'].iloc[-2] + optimal_solution_row[3] * \
                            df['demand'].iloc[-3])
    print(f"根据 QUBO 解预测的需求为: {predicted_demand_qubo}")

# 根据 AR(3) 模型预测的需求
predicted_demand_ar3 = model_fitted.params['const'] + model_fitted.params['demand.L1'] * df['demand'].iloc[-1] + \
                       model_fitted.params['demand.L2'] * df['demand'].iloc[-2] + model_fitted.params['demand.L3'] * \
                       df['demand'].iloc[-3]
print(f"根据 AR(3) 模型预测的需求为: {predicted_demand_ar3}")

#计算差异
if len(optimal_solution) > 0:
    difference = predicted_demand_ar3 - predicted_demand_qubo
    print(f"AR(3) 和 QUBO 解的预测差异为: {difference}")


plt.plot(df['demand'], label='Actual Demand')
plt.plot(range(len(df), len(df) + 1), predicted_demand_ar3, label='Forecasted Demand (AR(3))', marker='o', color='blue')
plt.plot(range(len(df), len(df) + 1), predicted_demand_qubo, label='Forecasted Demand (QUBO)', marker='x', color='red')


plt.legend()
plt.title("Demand Forecast: AR(3) vs QUBO")
plt.show()

#绘制AR(3)模型的残差图
plt.plot(model_fitted.resid)
plt.title('Residuals of AR(3) Model')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.show()

#绘制QUBO矩阵的热力图
plt.figure(figsize=(6, 5))
sns.heatmap(qubo_matrix_improved, annot=True, cmap='coolwarm', fmt='.2f', cbar=True)
plt.title("Heatmap of Improved QUBO Matrix")
plt.show()

ar3_differences = df['demand'] - predicted_demand_ar3
qubo_differences = df['demand'] - predicted_demand_qubo

#实际需求与预测需求的差异
plt.plot(ar3_differences, label='AR(3) Prediction Difference', color='blue')
plt.plot(qubo_differences, label='QUBO Prediction Difference', color='red')
plt.title('Prediction Differences (Actual vs Predicted)')
plt.xlabel('Time')
plt.ylabel('Difference')
plt.legend()
plt.show()

rolling_mean = df['demand'].rolling(window=3).mean()
rolling_std = df['demand'].rolling(window=3).std()

#滚动均值和标准差图
plt.plot(df['demand'], label='Actual Demand')
plt.plot(rolling_mean, label='Rolling Mean', color='green')
plt.plot(rolling_std, label='Rolling Std Dev', color='orange')
plt.title('Rolling Mean and Std Dev of Demand')
plt.legend()
plt.show()

#提取QUBO解中的权重
qubo_weights = np.array([np.sum(optimal_solution[:, i]) for i in range(4)])

#权重分布图
plt.bar(range(4), qubo_weights)
plt.title('Weight Distribution of QUBO Solution')
plt.xlabel('Coefficient Index')
plt.ylabel('Weight')
plt.show()

ar3_residuals = df['demand'] - predicted_demand_ar3
qubo_residuals = df['demand'] - predicted_demand_qubo

#绘制残差
plt.plot(ar3_residuals, label='AR(3) Residuals', color='blue')
plt.plot(qubo_residuals, label='QUBO Residuals', color='red')
plt.title('Residuals of AR(3) vs QUBO Predictions')
plt.xlabel('Time')
plt.ylabel('Residuals')
plt.legend()
plt.show()

