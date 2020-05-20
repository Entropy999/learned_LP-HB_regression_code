import argparse
import os
import torch
from data.ghg import getGHG
from data.gas import getGas
from data.electric import getElectric
from evaluate import *
from pathlib import Path
import sys
import copy
import time
import math
import random
from misc_utils import *
import warnings
import itertools
from tqdm import tqdm
import numpy as np
from sklearn.datasets import make_regression
from sklearn.linear_model import HuberRegressor, LinearRegression
from torch.autograd import Variable
from evaluate import *


def make_parser_reg():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data", type=str, default="gas", help="ghg|gas|electric")
   # aa("--dataname", type=str, default="mit", help="eagle|mit|friends")
    aa("--m", type=int, default=10, help="m for S")
    #aa("--iter", type=int, default=100, help="total iterations")
    aa("--iter", type=int, default=100000, help="total iterations")
    aa("--random", default=False, action='store_true',
       help="don't learn S! Just compute error on random S")

    aa("--size", type=int, default=2900, help="dataset size")
    #aa("--lr", type=float, default=1e-1, help="learning rate for gradient descent")
    aa("--lr", type=float, default=5e-3, help="learning rate for gradient descent")

    aa("--raw", dest='raw', default=True,
       action='store_true', help="generate raw?")
    aa("--bestonly", dest='bestonly', default=False,
       action='store_true', help="only compute best?")
    aa("--device", type=str, default="cuda:0")

    # aa("--n_sample_rows", type=int, default=-1, help="Train with n_sample_rows rows")
    aa("--k_sparse", type=int, default=1,
       help="number of values in a column of S, sketching mat")
    aa("--num_exp", type=int, default=1,
       help="number of times to rerun the experiment (for avg'ing results)")
    #aa("--bs", type=int, default=10, help="batch size")
    aa("--bs", type=int, default=32, help="batch size")

    aa("--initalg", type=str, default="random",
       help="random|kmeans|lev|gs|lev_cluster|load")
    aa("--load_file", type=str, default="",
       help="if initalg=load, where to get S?")

    aa("--save_fldr", type=str,
       help="folder to save experiment results into; if None, then general folder")  # default: None
    aa("--save_file", type=str, help="append to runtype, if not None")

    aa("--S_init_method", type=str, default="gaussian",
       help="pm1|gaussian|gaussian_pm1")
    aa("--greedy_number", type=int, default=3,
       help="the number of sampling")
    return parser


if __name__ == '__main__':
    runtype = "train_regression_huber"
    parser = make_parser_reg()
    args = parser.parse_args()
    rawdir = "/home/lynette"
    rltdir = "/home/lynette"
    print(args)
    m = args.m

    if args.data == 'ghg':
        save_dir_prefix = os.path.join(rltdir, "rlt", "ghg+huber")
    elif args.data == 'gas':
        save_dir_prefix = os.path.join(rltdir, "rlt", "gas+huber")
    elif args.data == 'electric':
        save_dir_prefix = os.path.join(rltdir, "rlt", "electric+huber")
    else:
        print("Wrong data option!")
        sys.exit()
    if args.save_file:
        runtype = runtype + "_" + args.save_file
    if args.save_fldr:
        save_dir = os.path.join(
            save_dir_prefix, args.save_fldr, args_to_fldrname(runtype, args))
    else:
        save_dir = os.path.join(
            save_dir_prefix, args_to_fldrname(runtype, args))
    best_fl_save_dir = os.path.join(save_dir_prefix, "best_files")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(best_fl_save_dir):
        os.makedirs(best_fl_save_dir)

    lr = args.lr  # default = 1
    if args.data == "ghg":
        AB_train, AB_test, n, d_a, d_b = getGHG(
            args.raw, args.size, rawdir, 100)
        A_train = AB_train[0]
        B_train = AB_train[1]
        A_test = AB_test[0]
        B_test = AB_test[1]
    elif args.data == "gas":
        AB_train, AB_test, n, d_a, d_b = getGas(args.raw, rawdir, 100)
        A_train = AB_train[0]
        B_train = AB_train[1]
        A_test = AB_test[0]
        B_test = AB_test[1]
    elif args.data == "electric":
        AB_train, AB_test, n, d_a, d_b = getElectric(args.raw, rawdir, 100)
        A_train = AB_train[0]
        B_train = AB_train[1]
        A_test = AB_test[0]
        B_test = AB_test[1]

    print("Working on data ", args.data)
    Path(save_dir).mkdir(parents=True, exist_ok=True)
    N_train = len(A_train)
    N_test = len(A_test)
    print("Dim= ", n, d_a, d_b)
    print("N train=", N_train, "N test=", N_test)
    p = 1.5
    best_file = os.path.join(best_fl_save_dir, "N=" + str(args.size) + '_best')
    if (not os.path.isfile(best_file)):
        print("computing best huber regression approximations")
        getbest_hb_regression(A_train, B_train, A_test, B_test, best_file)
    else:
        print("already got the best huber regression")
    best_train, best_test = torch.load(best_file)
    print("best train", best_train)
    print("best test", best_test)
    start = time.time()
    print_freq = 10  # TODO
    args_save_fpath = os.path.join(save_dir, "args_it_0.pkl")
    f = open(args_save_fpath, "wb")
    pickle.dump(vars(args), f)
    f.close()

    avg_over_exps = 0
    for exp_num in range(args.num_exp):

        it_save_dir = os.path.join(save_dir, "exp_%d" % exp_num)
        it_print_freq = print_freq
        it_lr = lr
        if not os.path.exists(it_save_dir):
            os.makedirs(it_save_dir)
        test_errs = []
        train_errs = []
        fp_times = []
        bp_times = []

        # Initialize sparsity pattern
        if args.initalg == "random":
            sketch_vector = torch.randint(m, [args.k_sparse, n]).int()
            print("-----------doing random-----------")
        elif args.initalg == "load":
            # TODO: not implemented for ksparse
            initalg = initalg_name2fn_dict[args.initalg]
            sketch_vector, sketch_value_cpu, active_ind = initalg(
                args.load_file, exp_num, n, m)
            sketch_value = sketch_value_cpu.detach()

        # Note: we sample with repeats, so you may map 1 row of A to <k_sparse distinct locations
        sketch_vector.requires_grad = False
        if args.initalg != "load":
            if args.S_init_method == "pm1":
                #sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(device)
                sketch_value = (
                    (torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).cpu()
            elif args.S_init_method == "gaussian":
                #sketch_value = torch.from_numpy(np.random.normal(size=[args.k_sparse, n]).astype("float32")).to(device)
                sketch_value = torch.from_numpy(np.random.normal(
                    size=[args.k_sparse, n]).astype("float32")).cpu()
            elif args.S_init_method == "gaussian_pm1":
                #sketch_value = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(device)
                sketch_value = (
                    (torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).cpu()
                #sketch_value = sketch_value + torch.from_numpy(np.random.normal(size=[args.k_sparse, n]).astype("float32")).to(device)
                sketch_value = sketch_value + torch.from_numpy(
                    np.random.normal(size=[args.k_sparse, n]).astype("float32")).cpu()


#######################zero_vector for Matrix stacking S#######################
        h = math.floor(math.log(n, 2))
        list = []
        for i in range(1, h):
            up_bound = math.floor(n/(pow(2, i)))
            list.append(torch.tensor(
                random.sample(range(0, n), int(up_bound))))

#######################zero_vector for Matrix stacking S########################


#################################D_Matrix####################################
        DMatrix_fl_save_dir = os.path.join(save_dir_prefix, "D_matrix")
        DMatrix_file = os.path.join(
            DMatrix_fl_save_dir, "N=" + str(args.size) + '_DMatrix')
        if not os.path.exists(DMatrix_fl_save_dir):
            os.makedirs(DMatrix_fl_save_dir)
        if (not os.path.isfile(DMatrix_file)):
            print("making D matrix")
            D = torch.zeros(
                h * S.shape[0], h * S.shape[0]).cpu()
            # h*S.shape[0], h*S.shape[0]).to(device)
            k = 0
            for i in range(D.shape[0]):
                for j in range(D.shape[1]):
                    if i == j:
                        D[i, j] = pow(2, k)
                        if (j+1) % m == 0:
                            k = k + 1
                        break
            torch.save([D], DMatrix_file)
        #D = torch.load(DMatrix_file)[0].to(device)
        D = torch.load(DMatrix_file)[0].cpu()

#################################D_Matrix####################################

############################evaluate random_sketch##############################

        batch_rand_ind = np.random.randint(0, high=N_train, size=args.bs)
        AM = A_train[batch_rand_ind].cpu()
        BM = B_train[batch_rand_ind].cpu()

        S = torch.zeros(m, n).cpu()
        S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(
            args.k_sparse)] = sketch_value.reshape(-1).cpu()
        S_add = torch.zeros(m, n).cpu()

        S_result = S+S_add
        for i in range(1, h):
            S_temp = S+S_add
        for j in range(len(list)):
        for k in range(m):
            S_temp[k][list[j]] = 0
        S_result = torch.cat([S_result, S_temp], dim=0)
        S_mul = torch.matmul(D, S_result)

        train_err, test_err = save_iteration_regression(
            S_mul, A_train, B_train, A_test, B_test, it_save_dir, bigstep, p)

############################evaluate random_sketch##############################

        np.save(os.path.join(it_save_dir, "train_errs.npy"),
                train_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "test_errs.npy"),
                test_errs, allow_pickle=True)

    print(avg_over_exps)
