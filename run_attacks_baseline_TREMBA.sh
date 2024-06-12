#!/bin/bash
 

python run_attacks_baseline_TREMBA.py  --attack_type B --victim swin_s --n_surrogates 10   --batch_size 1000 --pool 0 --device cuda:0 --pgd_iterations 100 --w_idx 0
