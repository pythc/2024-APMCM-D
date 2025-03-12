
# README

Registration Number: apmcm24206611  
University: Chongqing University of Posts and Telecommunications  
Category: Undergraduate Group  
From: Canyu Zhou  

## 1. Project Overview

This project aims to address the quantum computing and deep learning optimization problems in the 2024 Asia-Pacific Mathematical Contest in Modeling (APMCM). The project involves three main tasks, and quantum computing-related optimizations are performed using the Kaiwu SDK. This README file provides detailed information on the project structure, execution steps for tasks, installation of dependencies, and usage of datasets.

## 2. File Structure

```
├── Code/
│   ├── start.py              # Kaiwu SDK initialization
│   ├── task_1/               # Code for Task 1 
│   ├── task_2/               # Code for Task 2 
│   ├── task_3/               # Code for Task 3 
├── Data_set/
│   └── ml-100k/              # MovieLens 100k dataset
├── task1_figure/             # Figures for Task 1
├── task2_figure/             # Figures for Task 2
├── task3_figure/             # Figures for Task 3
├── requirements.txt          # List of dependencies 
└── README.md                 # Project description file
```

- `Code/`: Contains the main program files and code for each task.
- `Data_set/`: Contains the MovieLens 100k dataset used for Task 3.
- `task1_figure/`: Stores experimental result figures for Task 1.
- `task2_figure/`: Stores experimental result figures for Task 2.
- `task3_figure/`: Stores experimental result figures for Task 3.
- `requirements.txt`: List of required dependencies for environment setup.

## 3. Task Descriptions

### Task 1: Quantum Optimization Problem

- **File Path**: `Code/task_1/`
- **Description**: In Task 1, Kaiwu SDK is used to convert AR model expressions into a QUBO model. The simulated annealing algorithm from Kaiwu SDK is then applied to optimize the problem and improve prediction accuracy with quantum computing compatibility.

### Task 2: Demand Forecasting Model

- **File Path**: `Code/task_2/`
- **Description**: Task 2 transforms the optimization problem of an SVM classification model into a QUBO model. Both the SVM model and Kaiwu SDK are used to classify the dataset and compare results.

### Task 3: Movie Recommendation System

- **File Path**: `Code/task_3/`
- **Dataset Path**: `Data_set/ml-100k/`
- **Description**: In Task 3, a movie recommendation system is built using the MovieLens 100k dataset, with optimization conducted via the Kaiwu SDK.

## 4. How to Run

### Environment Setup

Before running the project, ensure that the required dependencies are installed. You can install them using the following command:

```bash
pip install -r requirements.txt
```

### Runing tasks

#### Task 1
1. Navigate to the Code/task_1/ directory.
2. Run the main program for Task 1:
   ```bash
   python task_1.py
   ```

#### Task 2
1. Navigate to the Code/task_2/ directory.
2. Run the main program for Task 2:
   ```bash
   python task_2.py
   ```

#### Task 3
1. Download and extract the MovieLens 100k dataset into the Data_set/ml-100k/
2. directory (skip this step if the dataset is already present).
3. Navigate to the Code/task_3/ directory.
   ```bash
   python task_3.py
   ```

## 5. Notes

- Ensure that the Python version is 3.8.
- Before running Task 3, make sure that the complete MovieLens 100k dataset is available in the Data_set/ml-100k/ directory.

## 6. Contact Information

 you have any questions or suggestions regarding this project, please contact the team member:
mail:[zhoucanyu2@outook.com]


## 1. 项目概述

本项目旨在解决2024年亚太地区数学建模竞赛（APMCM）中的量子计算和深度学习优化问题。项目涉及三个主要任务，并使用Kaiwu SDK进行量子计算相关的优化。本自述文件提供了项目结构、任务执行步骤、依赖安装和数据集使用的详细信息。

## 2. 文件结构

```
├── Code/
│   ├── start.py              # Kaiwu SDK 初始化
│   ├── task_1/               # 任务1的代码 
│   ├── task_2/               # 任务2的代码 
│   ├── task_3/               # 任务3的代码 
├── Data_set/
│   └── ml-100k/              # MovieLens 100k 数据集
├── task1_figure/             # 任务1的图表
├── task2_figure/             # 任务2的图表
├── task3_figure/             # 任务3的图表
├── requirements.txt          # 依赖列表 
└── README.md                 # 项目描述文件
```

- `Code/`: 包含主程序文件和各任务的代码。
- `Data_set/`: 包含用于任务3的MovieLens 100k数据集。
- `task1_figure/`: 存储任务1的实验结果图表。
- `task2_figure/`: 存储任务2的实验结果图表。
- `task3_figure/`: 存储任务3的实验结果图表。
- `requirements.txt`: 环境设置所需的依赖列表。

## 3. 任务描述

### 任务1: 量子优化问题

- **文件路径**: `Code/task_1/`
- **描述**: 在任务1中，使用Kaiwu SDK将AR模型表达式转换为QUBO模型。然后应用Kaiwu SDK中的模拟退火算法来优化问题，并通过量子计算兼容性提高预测精度。

### 任务2: 需求预测模型

- **文件路径**: `Code/task_2/`
- **描述**: 任务2将SVM分类模型的优化问题转化为QUBO模型。使用SVM模型和Kaiwu SDK对数据集进行分类并比较结果。

### 任务3: 电影推荐系统

- **文件路径**: `Code/task_3/`
- **数据集路径**: `Data_set/ml-100k/`
- **描述**: 在任务3中，使用MovieLens 100k数据集构建电影推荐系统，并通过Kaiwu SDK进行优化。

## 4. 如何运行

### 环境设置

在运行项目之前，请确保已安装所需的依赖项。您可以使用以下命令进行安装：

```bash
pip install -r requirements.txt
```

### 运行任务

#### 任务1
1. 导航到Code/task_1/目录。
2. 运行任务1的主程序：
   ```bash
   python task_1.py
   ```

#### 任务2
1. 导航到Code/task_2/目录。
2. 运行任务2的主程序：
   ```bash
   python task_2.py
   ```

#### 任务3
1. 下载并将MovieLens 100k数据集解压到Data_set/ml-100k/目录（如果数据集已存在，请跳过此步骤）。
2. 导航到Code/task_3/目录。
   ```bash
   python task_3.py
   ```

## 5. 注意事项

- 确保Python版本为3.8。
- 在运行任务3之前，请确保Data_set/ml-100k/目录中有完整的MovieLens 100k数据集。

## 6. 联系方式

如果您对本项目有任何问题或建议，请联系团队成员：
邮箱：[zhoucanyu2@outook.com]
