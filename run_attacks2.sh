#!/bin/bash
 
python run_attacks.py  --attack_type nPv0 --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --sw 20 --lmb 0.5 --eps 1 --device cuda:0 --pgd_iterations 200 &
python run_attacks.py  --attack_type nPv0 --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --sw 40 --lmb 0.5 --eps 1 --device cuda:0 --pgd_iterations 200 &

python run_attacks.py  --attack_type nPv0 --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --sw 40 --lmb 0 --eps 1 --device cuda:0 --pgd_iterations 200 &
python run_attacks.py  --attack_type nPv0 --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --sw 20 --lmb 0 --eps 1 --device cuda:0 --pgd_iterations 200 