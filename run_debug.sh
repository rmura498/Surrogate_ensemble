#!/bin/bash

python run_attacks.py  --attack_type A --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 0 --device cuda:2 --pgd_iterations 400 &
python run_attacks.py  --attack_type A --victim vgg19 --n_surrogates 10 --batch_size 100 --pool 1 --device cuda:2 --pgd_iterations 400


