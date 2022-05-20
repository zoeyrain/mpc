from copy import deepcopy
import time
import numpy as np
import pandas as pd
import pickle
from matplotlib import pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import SGDRegressor, RidgeCV
from sklearn.tree import DecisionTreeRegressor
from sklearn.metrics import make_scorer, mean_squared_error, r2_score

from mpc.tdpc import model
from mpc.tdpc import optimizer
from mpc.tdpc import simulator

def fit_function(x, his: np.ndarray, rule: np.ndarray):
    """calculate the output target

    Parameters
    ----------
    x (np.ndarray | np.matrix | list)
        The array that would be optimized. The dimension of this parameter is 
        (k, P), where k is the changing forecast horizen, P is the number of the
        control variables.
    his (list)
        The history control and target values. The dimension of this parameters
        is (r, n, 1), where r is the number of control rules, n is a data that
        decreases as the prediction step increases from N to 1, where N is the 
        sum of observe horizen and forecast horizen.
    rule (np.ndarray)
        The list of control rules, which dimension is (r, ).

    Returns
    -------
    yt (np.ndarray)
        The output targets
    """
    his = np.array(his).reshape((-1, 1))
    n = his.shape[0] + 1
    
    yt = np.zeros((rule[0].shape[0], 1))
    for i, r in enumerate(rule):
        r = r.reshape((-1, 1))
        yt[i] = np.dot(r[1:n].T, his) + r[0]
    
        if r.shape[0] > n:
            x = np.array(x).reshape((-1, 1))
            yt[i] += np.dot(r[n:].T, x)[0]
    return yt

def loss_function(x, his: list, rules: list, 
                  yref: list, W: np.matrix, R: np.matrix):
    """calculate the loss function

    Parameters
    ----------
    x (list | np.ndarray | np.matrix)
        The array that would be optimized. The dimension of this parameter is 
        (fh, P), where fh is the forecast horizen, P is the number of the control
        variables.
    his (list)
        The history control and target values. The dimension of this parameters
        is (r, n, 1), where r is the number of control rules, n is a data that
        decreases as the prediction step increases from N to 1, where N is the 
        sum of observe horizen and forecast horizen.
    rules (list):
        The list of control rules, which dimension is (r, ).
    yref (list)
        The reference of target values, which dimension is (P, 1)
    W (np.matrix)
        The matrix of the weights of the deviations between control values
        and reference values.
    Q (np.matrix)
        The matrix of the weights between the inputs and outputs.

    Returns
    -------
    loss (float)
        The value of loss function.
    """
    yref = np.array(yref)
    n_control = R.shape[1]
    x = np.matrix(x).reshape((-1, n_control))
    loss = 0
    for i, rule in enumerate(rules):
        h = his[i]
        yt = fit_function(x[:i+1], h, rule)
        
        dy = np.matrix(yt-yref)
        _loss = dy.T * W * dy 
        if yt < yref + 1:
            loss += _loss * 0.5
        else:
            loss += _loss

    x[1:] = np.diff(x.T).T
    x[0] = x[0] - his[0][-1,-1]
    loss += R * x.T * x
    return loss

if __name__ == "__main__":
    
    ## FIT MODEL
    weights = np.matrix(1)
    regulars = np.matrix([0.3])
    mse = make_scorer(mean_squared_error)
    
    estimator = RandomForestRegressor(n_estimators=100, max_depth=6, 
                                      max_features='sqrt')
    controller = RidgeCV(scoring=mse)
    opt = optimizer.Optimizer(ref=[48], W=weights, R=regulars, 
                           bounds=[(20.0, 50.0)], acq_func='EI', n_calls=50)

    mpc = model.RFDPC(lag=6, fh=4, targets=targets, dc=controls, dnc=features,
                estimator=estimator, controller=controller, optimizer=opt,
                loss_func=loss_function, fit_func=fit_function)
    """
    Parameters
    ----------

    lag (int)
        历史序列的观测长度，建议不要超过8
    fh (int)
        预测步长，建议不要大于lag
    target (list)
        被控参数的名称列表，如出口NOx值
    controls (list)
        控制参数的名称列表，例如阀门开度
    features (list)
        其他特征参数的名称列表，可调
    estimator = RandomForestRegressor(n_estimators=100, max_depth=6, 
                                      max_features='sqrt')
        估计器模型，必须为RandomForestRegressor，但其中的所有参数均可调
    controller = RidgeCV(scoring=mse)
        控制器模型，简单的线性模型即可，sklearn.linear_model中的模型均可
    optimizer = optimizer.Optimizer(ref=[48], W=weights, R=regulars, 
                           bounds=[(10.0, 60.0)], acq_func='EI', n_calls=50)
        优化器模型，其中ref为系统设定参数，W为target参数的权重，R为controls参数
        的权重，bounds为controls参数的上下限值，n_calls为优化器计算次数，越大计算时间越长
    fit_function
        拟合模型，不用改
    loss_function
        损失函数，由target，controls， ref，W 和 R 共同决定，其中W越大表示target部分的
        权重越大，输出更靠近设定值ref，R越大，表示controls部分的权重越大，系统控制变量
        controls的变化放缓，梯度下降
    """
    """
    建模过程
    1. 建立mpc模型 并使用历史数据拟合： mpc=xxxx, mpc.fit() （离线）
    2. 建立sim模型 并使用最新的数据拟合，sim=xxxx，sim.fit() （在线）
    3. 迭代控制，并更新数据： ut=sim.ut[-1], sim.update_fit() （在线）
    4. 若控制结果不理想，则回到第一步重新拟合mpc模型（更新数据/更新参数）
    """
    
    mpc.fit(X.iloc[:-100], y.iloc[:-100])
    ## pickle.dump(mpc, open("mpc.pickle.dat", "wb"))
    ## mpc = pickle.load(open("mpc.pickle.dat", "rb"))

    ## SIMULATOR
    sim = simulator.Simulator(mpc, controls=controls)
    sim.fit(X.iloc[:-100], y.iloc[:-100])
    dt = list()
    for i in range(20):
        xi = X.iloc[-100+i: -99+i]
        yi = y.iloc[-100+i: -99+i]
        t0 = time.time()
        sim.update_fit(xi, yi)
        dt.append(time.time()-t0)
    