#!/bin/bash

python run_attacks_baseline.py  --attack_type A --victim resnext50_32x4d --n_surrogates 10 --batch_size 100 --pool 0 --eps 1 --device cuda:0 --pgd_iterations 100 &
python run_attacks_baseline.py  --attack_type A --victim resnext50_32x4d --n_surrogates 20 --batch_size 100 --pool 0 --eps 1 --device cuda:2 --pgd_iterations 100 &

python run_attacks_baseline.py  --attack_type B --victim resnext50_32x4d --n_surrogates 10 --batch_size 100 --pool 0 --eps 1 --device cuda:1 --pgd_iterations 10 &
python run_attacks_baseline.py  --attack_type B --victim resnext50_32x4d --n_surrogates 20 --batch_size 100 --pool 0 --eps 1 --device cuda:2 --pgd_iterations 10 &

python run_attacks.py  --attack_type nPv0 --victim resnext50_32x4d --n_surrogates 10 --batch_size 100 --pool 0 --sw 40 --lmb 0.5 --eps 1 --device cuda:1 --pgd_iterations 100 &
python run_attacks.py  --attack_type nPv0 --victim resnext50_32x4d --n_surrogates 20 --batch_size 100 --pool 0 --sw 40 --lmb 0.5 --eps 1 --device cuda:2 --pgd_iterations 100 

