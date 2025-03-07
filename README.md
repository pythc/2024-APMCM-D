
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
