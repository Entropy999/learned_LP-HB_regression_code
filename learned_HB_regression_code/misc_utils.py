import numpy as np
import torch
import sys
import IPython
import os
import pickle
from evaluate import *
import warnings
import matplotlib.pyplot as plt
from global_variables import *
import math
import re
from sklearn.cluster import DBSCAN
import numpy_indexed as npi
from collections import Counter
# from evaluate import evaluate_to_rule_them_all_lp_regression


def args_to_fldrname(runtype, args):
    """
    :param args: from parse_args(), a namespace
    :return: str, foldername
    """
    ignore_keys = ["save_fldr", "save_file", "single", "bestonly", "bw", "dense", "dwnsmp",
                   "lr_S", "raw", "load_file", "device", "lev_count", "lev_cutoff", "lev_ridge"]
    d_args = vars(args)
    fldrname = runtype
    for key in sorted(d_args.keys()):
        if key not in ignore_keys:
            # print(key, d_args[key])
            fldrname += "_" + str(key) + "_" + str(d_args[key])
    # IPython.embed()
    return fldrname


def IRLS(yM, XM, maxiter, w_init=1, d=0.0001, tolerance=0.001):
    nsamples, nx, ny = XM.shape
    #X = XM.reshape((nsamples*nx, ny)).to(device)
    X = XM.reshape((nsamples*nx, ny)).cpu()
    del XM
    nsamples, nx, ny = yM.shape
    #y = yM.reshape((nsamples*nx, ny)).to(device)
    y = yM.reshape((nsamples*nx, ny)).cpu()
    del yM

    n, p = X.shape
    delta_ = np.array(np.repeat(d, n)).reshape(1, n)
    delta = torch.from_numpy(delta_).cpu()
    w = np.repeat(1, n)
    #W = torch.from_numpy(np.diag(w)).to(device)
    W = torch.from_numpy(np.diag(w)).cpu()


    temp1 = torch.matmul(torch.t(X), W.float())
    para1 = torch.inverse(torch.matmul(temp1, X))
    temp2 = torch.matmul(torch.t(X), W.float())
    para2 = torch.matmul(temp2, y)
    #B = torch.matmul(para1, para2).to(device)
    B = torch.matmul(para1, para2).cpu()

    for _ in range(maxiter):
        #B_add = torch.zeros(B.shape[0]).to(device)
        B_add = torch.zeros(B.shape[0], B.shape[1]).cpu()

        _B = B + B_add
        #_w = torch.t(abs(y - torch.matmul(X, B.float()))).to(device)
        _w = torch.t(abs(y - torch.matmul(X, B.float()))).cpu()

        _w = _w.to(torch.float32)
        #delta = delta.to(torch.float32).to(device)
        delta = delta.to(torch.float32).cpu()

        w = float(1)/torch.max(delta, _w)
        W = torch.diag(w[0])
        temp1 = torch.matmul(torch.t(X).double(), W.double())
        para1 = torch.inverse(torch.matmul(temp1, X.double()))
        temp2 = torch.matmul(torch.t(X).double(), W.double())
        para2 = torch.matmul(temp2, y.double())
        B = torch.matmul(para1, para2)
        crit = torch.nn.SmoothL1Loss()
        loss = crit(_B.float(), B.float())
        tol = torch.sum(loss)
        # print("Tolerance = %s" % tol)
        if tol < tolerance:
            return B
    return B


def huber_regression(A, B):
    X = IRLS(yM=B, XM=A, maxiter=50)

    return X
