import json
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn import linear_model
from sklearn.model_selection import KFold
from sklearn.metrics import adjusted_rand_score
from sklearn.metrics import normalized_mutual_info_score
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
from sklearn import svm
import torch



# for regression tasks

def compute_metrics(y_pred, y_test):
    y_pred[y_pred<0] = 0
    mae = mean_absolute_error(y_test, y_pred)
    mse = mean_squared_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    return mae, np.sqrt(mse), r2


def regression(X_train, y_train, X_test, alpha):
    reg = linear_model.Ridge(alpha=alpha)
    
    X_train = np.array(X_train, dtype=float)
    y_train = np.array(y_train, dtype=float)
    reg.fit(X_train, y_train)

    y_pred = reg.predict(X_test)
    return y_pred

def kf_predict(X, Y):

    kf = KFold(n_splits=5)
    y_preds = []
    y_truths = []
    for train_index, test_index in kf.split(X):
        X_train, X_test = X[train_index], X[test_index]
        y_train, y_test = Y[train_index], Y[test_index]
        y_pred = regression(X_train, y_train, X_test, 1)
        y_preds.append(y_pred)
        y_truths.append(y_test)

    return np.concatenate(y_preds), np.concatenate(y_truths)


def predict_regression(embs, labels, display=False):
    
    y_pred, y_test = kf_predict(embs, labels)
    mae, rmse, r2 = compute_metrics(y_pred, y_test)
    if display:
        print("MAE: ", mae)
        print("RMSE: ", rmse)
        print("R2: ", r2)
    return mae, rmse, r2






def adj_space(vid, path='path/adj_for_space.pkl'):
    # 读取数据
    df_voroni = pd.read_pickle(path)
    com = []

    # 处理每个街道
    for item in list(vid):
        df = df_voroni[df_voroni['street'] == item]
        disct = list(df.ID)  # 假设 'ID' 是你需要的列名
        com.extend(disct)

    # 创建邻接矩阵
    df_a = df_voroni.loc[com, com]
    disct_m = np.array(df_a.values)
    disct_m = disct_m / 1000

    return disct_m, com