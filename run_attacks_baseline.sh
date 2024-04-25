#!/bin/bash
 

python run_attacks.py  --attack_type B --victim vgg19 --n_surrogates 10   --batch_size 100 --pool 0 --device cuda:2 --pgd_iterations 10 &
python run_attacks.py  --attack_type B --victim vgg19 --n_surrogates 10  --batch_size 100 --pool 2 --device cuda:2 --pgd_iterations 10 &
python run_attacks.py --attack_type nP --victim vgg19 --n_surrogates 10  --batch_size 100 --pool 2 --device cuda:2 --pgd_iterations 10 &
python run_attacks.py  --attack_type nP --victim vgg19 --n_surrogates 10  --batch_size 100 --pool 0 --device cuda:2 --pgd_iterations 10 &

python run_attacks.py  --attack_type A --victim vgg19 --n_surrogates 10  --batch_size 100 --pool 0 --device cuda:0 --pgd_iterations 300 &
python run_attacks.py  --attack_type A --victim vgg19 --n_surrogates 10  --batch_size 100 --pool 2 --device cuda:0 --pgd_iterations 300 &

python run_attacks.py  --attack_type nP --victim vgg19 --n_surrogates 10  --batch_size 100 --pool 0 --device cuda:0 --pgd_iterations 50 &
python run_attacks.py  --attack_type nP --victim vgg19 --n_surrogates 10  --batch_size 100 --pool 2 --device cuda:2 --pgd_iterations 50 