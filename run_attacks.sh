#!/bin/bash
 
python run_attacks.py  --attack_type nP --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 0 --device cuda:0 --pgd_iterations 10 &
python run_attacks.py  --attack_type nP --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 1 --device cuda:1 --pgd_iterations 10 &


python run_attacks.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 0 --device cuda:2 --pgd_iterations 10 &
python run_attacks.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 1 --device cuda:0 --pgd_iterations 10 &

python run_attacks.py  --attack_type nP --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 0 --device cuda:0 --pgd_iterations 100 &
python run_attacks.py  --attack_type nP --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 1 --device cuda:1 --pgd_iterations 100 &


python run_attacks.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 0 --device cuda:2 --pgd_iterations 100 &
python run_attacks.py  --attack_type B --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 1 --device cuda:0 --pgd_iterations 100 
