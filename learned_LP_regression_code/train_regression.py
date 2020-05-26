import argparse
import os
import torch
from data.ghg import getGHG
from data.gas import getGas
from data.electric import getElectric
from evaluate import *
#from torchviz import make_dot, make_dot_from_trace
from pathlib import Path
import sys
import time
import math
import random
from misc_utils import *
import warnings
from tqdm import tqdm
import numpy as np

from scipy.stats import levy_stable


def make_parser_reg():
    parser = argparse.ArgumentParser()

    def aa(*args, **kwargs):
        parser.add_argument(*args, **kwargs)

    aa("--data", type=str, default="gas", help="ghg|gas|electric")
   # aa("--dataname", type=str, default="mit", help="eagle|mit|friends")
    aa("--m", type=int, default=10, help="m for S")
    aa("--iter", type=int, default=10000, help="total iterations")
    # aa("--scale", type=int, default= 100, help="scale") # not a functioning argument, assume 100

    aa("--random", default=False, action='store_true',
       help="don't learn S! Just compute error on random S")

    aa("--size", type=int, default=2900, help="dataset size")
    aa("--lr", type=float, default=5e-2, help="learning rate for gradient descent")
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
    aa("--bs", type=int, default=64, help="batch size")
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
    runtype = "train_regression_lp"
    parser = make_parser_reg()
    args = parser.parse_args()
    rawdir = "/home/lynette"
    rltdir = "/home/lynette"
    print(args)
    m = args.m

    if args.data == 'ghg':
        save_dir_prefix = os.path.join(rltdir, "rlt", "ghg+LP")
        # print("---------------testing1-----------")
        # TODO
    elif args.data == 'gas':
        save_dir_prefix = os.path.join(rltdir, "rlt", "gas+LP")
    elif args.data == 'electric':
        save_dir_prefix = os.path.join(rltdir, "rlt", "electric+LP")
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
    # print("---------------testing2-----------")
    best_fl_save_dir = os.path.join(save_dir_prefix, "best_files")
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    if not os.path.exists(best_fl_save_dir):
        os.makedirs(best_fl_save_dir)

    if (not args.bestonly) and (len(os.listdir(save_dir))):
        print("This experiment is already done! Now exiting.")
        # sys.exit()
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

    # save args

    avg_over_exps = 0
    for exp_num in range(args.num_exp):

        it_save_dir = os.path.join(save_dir, "exp_%d" % exp_num)
        it_print_freq = print_freq
        it_lr = lr

        if not os.path.exists(it_save_dir):
            os.makedirs(it_save_dir)

        test_errs = []
        train_errs = []
        #device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
        #device = torch.device("cpu")
        # Initialize sparsity pattern
        if args.initalg == "random":
            sketch_vector = torch.randint(m, [args.k_sparse, n]).int()
            sketch_vector = torch.randint(m, [args.k_sparse, n]).int()
            print()
            print("-----------doing random-----------")
        elif args.initalg == "load":
            # TODO: not implemented for ksparse
            initalg = initalg_name2fn_dict[args.initalg]
            sketch_vector, sketch_value_cpu, active_ind = initalg(
                args.load_file, exp_num, n, m)
            sketch_value = sketch_value_cpu.detach().cpu()

        # Note: we sample with repeats, so you may map 1 row of A to <k_sparse distinct locations
        sketch_vector.requires_grad = False
        if args.initalg != "load":
            if args.S_init_method == "pm1":
                sketch_value = (
                    (torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).cpu()
                # print("-------pm1----sketch_value--------")
                # print(sketch_value.size)
            elif args.S_init_method == "gaussian":
                sketch_value = torch.from_numpy(levy_stable.rvs(
                    p, 0, size=[args.k_sparse, n]).astype("float32")).cpu()
                # sketch_value_cpu = ((torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).to(args.device)
            elif args.S_init_method == "gaussian_pm1":
                sketch_value = (
                    (torch.randint(2, [args.k_sparse, n]).float() - 0.5) * 2).cpu()
                sketch_value = sketch_value + torch.from_numpy(
                    np.random.normal(size=[args.k_sparse, n]).astype("float32")).cpu()

############################evaluate random_sketch##############################
        S = torch.zeros(m, n).cpu()

        S[sketch_vector.type(torch.LongTensor).reshape(-1), torch.arange(n).repeat(
            args.k_sparse)] = sketch_value.reshape(-1).cpu()

        train_err, test_err = save_iteration_regression(
            S, A_train, B_train, A_test, B_test, it_save_dir, bigstep, p)

############################evaluate random_sketch##############################

        np.save(os.path.join(it_save_dir, "train_errs.npy"),
                train_errs, allow_pickle=True)
        np.save(os.path.join(it_save_dir, "test_errs.npy"),
                test_errs, allow_pickle=True)
    print(avg_over_exps)
