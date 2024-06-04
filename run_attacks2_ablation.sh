#!/bin/bash

python run_attacks_ablation_alpha.py  --attack_type nPF --loss CW --victim swin_s --n_surrogates 10 --batch_size 100 --pool 2 --sw 40 --lmb 0 --eps 0 --device cuda:0 --pgd_iterations 100
#python run_attacks.py  --attack_type nPF --loss CW --victim vgg19 --n_surrogates 10 --batch_size 1000 --pool 2 --sw 40 --lmb 0.5 --eps 1 --device cuda:0 --pgd_iterations 100

#python run_attacks.py  --attack_type nPF --loss CW --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 0 --sw 40 --lmb 0.5 --eps 0 --device cuda:2 --pgd_iterations 500 &
#python run_attacks.py  --attack_type nPF --loss CW --victim vgg19 --n_surrogates 20 --batch_size 100 --pool 0 --sw 40 --lmb 0.5 --eps 0 --device cuda:0 --pgd_iterations 500
