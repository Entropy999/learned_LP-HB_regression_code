#python train_regression.py --data ghg --m 20 --size 500 --bestonly --num_exp 1 --bs 20 --lr 10.0 --iter 1000

#python train_regression.py --data gas --m 20 --bs 20 --lr 50.0 --iter 1000 --num_exp 1 --bestonly

#python train_regression.py --data electric --m 20 --bs 20 --lr 20.0 --iter 1000 --num_exp 1 --bestonly
#python train_regression.py --data electric --m 20 --bs 20 --lr 30.0 --iter 1000 --num_exp 1 --bestonly
python -u train_regression.py --data electric --m 10 --save_fldr regression_ablation --device cuda:0 --bs 20 --lr 20.000000 --iter 1000 --bestonly
python -u train_regression.py --data electric --m 10 --save_fldr regression_ablation --device cuda:0 --bs 20 --lr 30.000000 --iter 1000 --bestonly