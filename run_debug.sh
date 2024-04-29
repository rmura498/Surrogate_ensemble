#!/bin/bash


python run_attacks_baseline.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --eps 1 --device cuda:0 --pgd_iterations 10 &
python run_attacks_baseline.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --eps 1 --mul 2 --device cuda:0 --pgd_iterations 10 &

python run_attacks_baseline.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --eps 0 --device cuda:0 --pgd_iterations 10 &
python run_attacks_baseline.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 0 --eps 0 --device cuda:0 --pgd_iterations 10 &

python run_attacks_baseline.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --eps 1 --device cuda:2 --pgd_iterations 10 &
python run_attacks_baseline.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --eps 1 --mul 2 --device cuda:2 --pgd_iterations 10 &

python run_attacks_baseline.py  --attack_type B --victim swin_s --n_surrogates 10 --batch_size 100 --pool 2 --eps 0 --device cuda:2 --pgd_iterations 10 &

python run_attacks_baseline.py  --attack_type A --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --eps 0 --device cuda:0 --pgd_iterations 500 &
python run_attacks_baseline.py  --attack_type A --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 2 --eps 1 --device cuda:1 --pgd_iterations 500 &
python run_attacks_baseline.py  --attack_type A --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 0 --eps 0 --device cuda:2 --pgd_iterations 500 &
python run_attacks_baseline.py  --attack_type A --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 0 --eps 1 --device cuda:2 --pgd_iterations 500 