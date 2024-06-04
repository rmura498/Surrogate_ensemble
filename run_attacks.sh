#!/bin/bash

python run_attacks_baseline.py  --attack_type A --loss CW --victim swin_s --n_surrogates 10 --batch_size 100 --pool 0 --eps 0 --device cuda:1 --pgd_iterations 100 &
python run_attacks_baseline.py  --attack_type A --loss CW --victim swin_s --n_surrogates 10 --batch_size 100 --pool 0 --eps 1 --device cuda:1 --pgd_iterations 100 &

#python run_attacks_baseline.py  --attack_type A --loss CW --victim vit_l_16 --n_surrogates 10 --batch_size 100 --pool 0 --eps 0 --device cuda:1 --pgd_iterations 100 &
#python run_attacks_baseline.py  --attack_type A --loss CW --victim vit_l_16 --n_surrogates 10 --batch_size 100 --pool 0 --eps 1 --device cuda:1 --pgd_iterations 100
