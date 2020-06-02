import torch
import numpy as np
import sys
from global_variables import *
import IPython
import math
import os
from misc_utils import *


def save_iteration_regression(S, A_train, B_train, A_test, B_test, save_dir, bigstep, p):
    """
    Not implemented:
    Mixed matrix evaluation
    """
    torch_save_fpath = os.path.join(save_dir, "it_%d" % bigstep)

    test_err = evaluate_to_rule_them_all_lp_regression(A_test, B_test, S, p)
    train_err = evaluate_to_rule_them_all_lp_regression(
        A_train, B_train, S, p)
    torch.save([[S], [train_err, test_err]], torch_save_fpath)

    print(train_err, test_err)
    print("Saved iteration: %d" % bigstep)
    return train_err, test_err


def evaluate_to_rule_them_all_lp_regression(A_set, B_set, S, p):
    n = A_set.size()[0]
    bs = 100
    loss = 0
    # median = 0
    for i in range(math.ceil(n / float(bs))):
        AM = A_set[i * bs:min(n, (i + 1) * bs)].cpu()
        BM = B_set[i * bs:min(n, (i + 1) * bs)].cpu()
        SA = torch.matmul(S.cpu(), AM)
        SB = torch.matmul(S.cpu(), BM)
        X = lp_norm_regression(p, SA, SB).cpu()
        ans = AM.matmul(X.float())
        it_loss = torch.sum(torch.norm(abs(ans - BM), dim=(1, 2), p=p))//n
        loss += it_loss.item()
    return loss


def bestPossible_lp_regression(A_set, B_set, p):
    n = A_set.size()[0]
    bs = 100  # 100
    loss = 0
    median = 0
    A_set = A_set.cpu()
    B_set = B_set.cpu()
    for i in range(math.ceil(n / float(bs))):
        AM = A_set[i * bs:min(n, (i + 1) * bs)].cpu()
        BM = B_set[i * bs:min(n, (i + 1) * bs)].cpu()
        X = lp_norm_regression(p, AM, BM).cpu()
        ans = AM.matmul(X.float())
        it_loss = torch.sum(torch.norm(abs(ans - BM), dim=(1, 2), p=p))//n
        loss += it_loss.item()
    print("======================LP regression===============")
    print(loss)
    print("======================LP regression===============")
    return loss


def getbest_regression(A_train, B_train, A_test, B_test, best_file):
    best_train_err = bestPossible_regression(A_train, B_train)
    best_test_err = bestPossible_regression(A_test, B_test)

    torch.save([best_train_err, best_test_err], best_file)
    return best_train_err, best_test_err


def getbest_lp_regression(A_train, B_train, A_test, B_test, best_file, p):
    best_train_err = bestPossible_lp_regression(A_train, B_train, p)
    best_test_err = bestPossible_lp_regression(A_test, B_test, p)

    torch.save([best_train_err, best_test_err], best_file)
    return best_train_err, best_test_err
