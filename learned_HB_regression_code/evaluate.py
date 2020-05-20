import torch
import numpy as np
import sys
from global_variables import *
import IPython
import math
from misc_utils import *
import os
import pickle


def save_iteration_regression(S, A_train, B_train, A_test, B_test, save_dir, bigstep, p):
    """
    Not implemented:
    Mixed matrix evaluation
    """
    torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)
    test_err = evaluate_to_rule_them_all_huber_regression(A_test, B_test, S)
    train_err = evaluate_to_rule_them_all_huber_regression(
        A_train, B_train, S)
    torch.save([[S], [train_err, test_err]], torch_save_fpath)

    print("random_sketch: train_error and test_error:")
    print(train_err, test_err)
    return train_err, test_err


def evaluate_to_rule_them_all_huber_regression(A_set, B_set, S):
    n = A_set.size()[0]
    bs = 100
    loss = 0
    median = 0
    for i in range(math.ceil(n/float(bs))):
        #AM = A_set[i*bs:min(n, (i+1)*bs)].to(device)
        AM = A_set[i*bs:min(n, (i+1)*bs)].cpu()
        #BM = B_set[i*bs:min(n, (i+1)*bs)].to(device)
        BM = B_set[i*bs:min(n, (i+1)*bs)].cpu()

        #SA = torch.matmul(S.to(device), AM)
        SA = torch.matmul(S.cpu(), AM)

        #SB = torch.matmul(S.to(device), BM)
        SB = torch.matmul(S.cpu(), BM)

        #X = huber_regression(SA, SB)
        X = huber_regression(SA, SB).cpu()

        ans = AM.matmul(X.float())
        loss = abs(ans-BM)
        #median += np.median(loss.detach().numpy())
        median += np.median(loss.detach().cpu().numpy())
    median = median/math.ceil(n/float(bs))
    return median
