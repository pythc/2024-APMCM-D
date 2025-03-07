import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, confusion_matrix, precision_recall_curve, f1_score
import seaborn as sns
import kaiwu as kw


iris = datasets.load_iris()
X = iris.data
y = iris.target
iris_df = pd.DataFrame(X, columns=iris.feature_names)
iris_df['target'] = y
print(f"Dataset size: {X.shape[0]} samples, {X.shape[1]} features")

#划分为训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

#进行标准化处理
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

#训练支持向量机
svm_model = SVC(kernel='linear')  # 创建一个线性核的SVM模型
svm_model.fit(X_train, y_train)
print("SVM model training completed.")

#进行预测并计算准确率
y_pred = svm_model.predict(X_test)
svm_accuracy = accuracy_score(y_test, y_pred)
print(f"SVM model test accuracy: {svm_accuracy:.4f}")

#散点图
plt.figure(figsize=(10, 8))
sns.pairplot(iris_df, hue="target", markers=["o", "s", "D"])
plt.suptitle("Scatter Plot: Iris Dataset Feature Distribution", y=1.02)
plt.show()

#小提琴图
plt.figure(figsize=(12, 8))
sns.violinplot(x="target", y="sepal length (cm)", data=iris_df)
plt.title("Violin Plot: Sepal Length Distribution by Class")
plt.show()

#混淆矩阵
cm = confusion_matrix(y_test, y_pred)
plt.figure(figsize=(8, 6))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", xticklabels=iris.target_names, yticklabels=iris.target_names)
plt.title("Confusion Matrix: SVM Classification Results")
plt.ylabel('真实类别标签')
plt.xlabel('预测类别标签')
plt.show()

#精确率-召回率曲线
precision, recall, _ = precision_recall_curve(y_test, y_pred, pos_label=1)
plt.figure(figsize=(8, 6))
plt.plot(recall, precision, marker='.')
plt.title("Precision-Recall Curve")
plt.xlabel("召回率")
plt.ylabel("精确率")
plt.show()

#绘制F1分数曲线
f1 = f1_score(y_test, y_pred, average=None)
plt.figure(figsize=(8, 6))
plt.plot(f1, marker='.')
plt.title("F1 Score Curve")
plt.xlabel("类别")
plt.ylabel("F1分数")
plt.show()

#线性核的决策边界
plt.figure(figsize=(10, 8))
X_train_2d = X_train[:, :2]  #选取前两个特征用于二维可视化
svm_model_linear = SVC(kernel='linear')
svm_model_linear.fit(X_train_2d, y_train)
xx, yy = np.meshgrid(np.linspace(X_train_2d[:, 0].min(), X_train_2d[:, 0].max(), 100),
                     np.linspace(X_train_2d[:, 1].min(), X_train_2d[:, 1].max(), 100))
Z = svm_model_linear.predict(np.c_[xx.ravel(), yy.ravel()])
Z = Z.reshape(xx.shape)
plt.contourf(xx, yy, Z, alpha=0.8)
plt.scatter(X_train_2d[:, 0], X_train_2d[:, 1], c=y_train, edgecolors='k', marker='o')
plt.title("Decision Boundary with Linear Kernel")
plt.xlabel("sepal length")
plt.ylabel("sepal width")
plt.show()

#从SVM模型推导QUBO矩阵
svm_coef = svm_model.coef_
svm_intercept = svm_model.intercept_

#确保svm_intercept为标量，而非数组
if len(svm_intercept) == 1:
    svm_intercept = svm_intercept[0]

if len(svm_coef) > 1:
    svm_coef = svm_coef[0]

#样本数量
n_samples = X_train.shape[0]

alpha = np.zeros(n_samples)
C = 1.0
for i in range(n_samples):
    for j in range(n_samples):
        alpha[i] += alpha[j] * y[i] * y[j] * np.dot(X[i], X[j])
    alpha[i] -= y[i] * (svm_intercept + np.dot(svm_coef, X[i]))
    alpha[i] += C * (y[i]) **2

binary_alpha = np.where(alpha >= 0, 1, 0)

Q = np.zeros((n_samples, n_samples))
for i in range(n_samples):
    for j in range(n_samples):
        Q[i][j] = alpha[i] * alpha[j] * y[i] * y[j] * np.dot(X[i], X[j])

#模拟退火优化
output = kw.classical.SimulatedAnnealingOptimizer(
    initial_temperature=100,
    alpha=0.99,
    cutoff_temperature=0.001,
    iterations_per_t=100,
    size_limit=10
).solve(Q)
print(f"Simulated Annealing solution vector: {output}")

qubo_predicted_y = []
for solution in output:
    prediction = np.dot(solution, binary_alpha)
    qubo_predicted_y.append(prediction)

qubo_accuracy = accuracy_score(y_test, qubo_predicted_y)
print(f"QUBO model prediction accuracy: {qubo_accuracy:.4f}")