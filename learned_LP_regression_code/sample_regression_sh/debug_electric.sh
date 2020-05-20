# bug: for larger m, larger error
python -u train_regression.py --data electric --m 10 --num_exp 1 --save_fldr regression_ablation --device cuda:1 --bs 20 --lr 30.000000 --iter 1000 --bestonly --random --raw
python -u train_regression.py --data electric --m 20 --num_exp 1 --save_fldr regression_ablation --device cuda:1 --bs 20 --lr 30.000000 --iter 1000 --bestonly --random
python -u train_regression.py --data electric --m 30 --num_exp 1 --save_fldr regression_ablation --device cuda:1 --bs 20 --lr 30.000000 --iter 1000 --bestonly --random