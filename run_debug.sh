#!/bin/bash

python run_attacks.py  --attack_type nPF --loss CW --victim vgg19 --n_surrogates 5 --batch_size 100 --pool 2 --sw 40 --lmb 0.5 --eps 0 --device cuda:2 --pgd_iterations 100 &
python run_attacks.py  --attack_type nPF --loss CW --victim vgg19 --n_surrogates 5 --batch_size 100 --pool 0 --sw 40 --lmb 0.5 --eps 0 --device cuda:1 --pgd_iterations 200 &
