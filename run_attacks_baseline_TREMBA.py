import torch
import torch.nn as nn
import json

from torch.utils.data import DataLoader
from Attack.BaselineTREMBA import BaselineTREMBA
from Utils.TREMBA_Utils import Normalize, utils
from config import SURROGATE_NAMES
from Utils.utils import save_json, generate_time
from Utils.load_models import load_dataset, load_model
import argparse
import os
import numpy as np

torch.manual_seed(0)

parser = argparse.ArgumentParser(description="Run Attacks")
parser.add_argument('--config', default='config_TREMBA.json', help='config file')
parser.add_argument('--save_prefix', default=None, help='override save_prefix in config file')
parser.add_argument('--model_name', default=None)
parser.add_argument('--attack_type', type=str, default='B', choices=['B', 'A'], help='Type of attack')
parser.add_argument('--victim', type=str, default='vgg19',
                    choices=['resnext50_32x4d', 'vgg19', 'densenet121', 'alexnet', 'swin_s', 'shufflenet_v2_x2_0',
                             'regnet_y_32gf', 'efficientnet_v2_l', 'vit_l_16'], help='Type of attack')
parser.add_argument('--n_surrogates', type=int, default=20, help='Number of Surrogates')
parser.add_argument('--batch_size', type=int, default=100, help='Number of sample to evaluate')
parser.add_argument('--device', type=str, default='cuda:0', choices=['cuda:0', 'cuda:1', 'cuda:2', 'cpu'], help='Device to use (cpu, cuda:0, cuda:1)')
parser.add_argument('--attack_iterations', type=int, default=40, help='Number of attack iterations')
parser.add_argument('--pgd_iterations', type=int, default=10, help='Number of pgd iterations')
parser.add_argument('--loss', type=str, default='CW', choices=['CW', 'CE'], help='Loss function')
parser.add_argument('--pool', type=str, default='0', choices=['0', '1', '2'], help='Pool of surrogates')
parser.add_argument('--eps', type=str, default='0', help='Perturbation Size, 0 16/255, 1 8/255 2 4/255')
parser.add_argument('--mul', type=int, default=1, help='multiplier step')
parser.add_argument('--w_idx', type=int, default=0, help='index of generator weight file')
args = parser.parse_args()

multiplier = int(args.mul)

eps_dict = {'0': [16/255, 1], 
            '1': [8/255, 1],
            '2': [4/255, 2]}
tm = eps_dict[args.eps]

# attacks parameters
numb_surrogates = int(args.n_surrogates)
batch_size = int(args.batch_size)
device = args.device
loss = args.loss
attack_iterations = int(args.attack_iterations)
pgd_iterations = int(args.pgd_iterations)
attack_type = args.attack_type
victim_name = args.victim
eps = float(tm[0])
lr_w = 5e-2
alpha = tm[1] * multiplier * 3 * eps / 10
x = alpha
pool = int(args.pool)

attack_dict = {'B': BaselineTREMBA}
attack = attack_dict[attack_type]

if pool == 0:
    surrogates = [surr for surr in SURROGATE_NAMES[:numb_surrogates]]
elif pool == 1:

    surrogates_pool = ["densenet161", "efficientnet_v2_l", "regnet_y_16gf",
                  "resnet101", 'inception_v3', 'mnasnet1_0', 'googlenet',
                  'resnet18', 'convnext_small', 'mobilenet_v3_small']

    surrogates = [surr for surr in surrogates_pool[:numb_surrogates]]

elif pool == 2:
    surrogates_pool =['inception_v3', 'mobilenet_v3_small', 'squeezenet1_1', 'googlenet', 'resnet18',
                  'mnasnet1_0', 'densenet161', 'efficientnet_b0',
                  'regnet_y_400mf', 'resnext101_32x8d', 'resnet152_denoise', 'resnet101_denoise']
    surrogates = [surr for surr in surrogates_pool[:numb_surrogates]]

with open(args.config) as config_file:
    state = json.load(config_file)

if args.save_prefix is not None:
    state['save_prefix'] = args.save_prefix
if args.model_name is not None:
    state['model_name'] = args.model_name

new_state = state.copy()
new_state['batch_size'] = 1
new_state['test_bs'] = 1
device = torch.device(args.device if torch.cuda.is_available() else "cpu")

weight_idx = ['0', '20', '40', '60', '80', '100']

weight = torch.load(os.path.join("G_weight", state['generator_name'] + '_' + weight_idx[args.w_idx] + ".pytorch"), map_location=device)

encoder_weight = {}
decoder_weight = {}
for key, val in weight.items():
    if key.startswith('0.'):
        encoder_weight[key[2:]] = val
    elif key.startswith('1.'):
        decoder_weight[key[2:]] = val


# for imagenet1000
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])
nlabels = 1000

# loading dataset
dataset = load_dataset(dataset_name='imagenet')
dataloader = DataLoader(
    dataset=dataset,
    batch_size=batch_size,
    shuffle=False
)
images, labels, targets = next(iter(dataloader))
images = images.to(device)
labels = labels.to(device)
targets = targets.to(device)


# load models
victim_model = load_model(victim_name, device=device).to(device)

if 'defense' in state and state['defense']:
    model = nn.Sequential(
        Normalize.Normalize(mean, std),
        Normalize.Permute([2,1,0]),
        victim_model
    )
else:
    model = nn.Sequential(
        Normalize.Normalize(mean, std),
        victim_model
    )


def attack_evaluate():
    global state
    # global attack, attack_id, surrogates
    # global numb_surrogates, eps, pgd_iterations, lr_w, attack_iterations, alpha, x, batch_size, loss
    # global victim_model, ens_surrogates, victim_name
    # global images, labels, targets

    results_dict = {}
    print(eps)

    results_dict['ensemble'] = surrogates
    # print("Surrogates", len(ens_surrogates))
    # instantiate attack

    function = utils.Function(victim_model, state['batch_size'], state['margin'], nlabels, True)
    attacker = attack(function, encoder_weight, decoder_weight, victim_model, attack_iterations,
                      alpha, lr_w, eps, pgd_iterations, loss=loss, device=device)
    state['target_class'] = int(weight_idx[args.w_idx])
    # for idx in range(batch_size):
    for idx in range(batch_size):
        image = images[idx]
        label = labels[idx]
        # target = targets[idx]
        # state['target_class'] = target
        # function = utils.Function(victim_model, state['batch_size'], state['margin'], nlabels, state['target'])
        print(f"\n-------- Sample Number:{idx} - victim {victim_name} -------- ")
        print(f"### {str(attack_dict[attack_type].__name__)} ###")
        query_b, loss_list_b, n_iter_b = attacker.forward(image, label, state)
        results_dict[f'{idx}'] = {'query': query_b,
                                  'loss_list': loss_list_b,
                                  'n_iter': n_iter_b}

    save_json(results_dict,f'{generate_time()}_{str(attack_dict[attack_type].__name__)}_{victim_name}_b{batch_size}_eps{str(eps)[0:5]}_alp{str(alpha)[0:5]}_pool{pool}_surr{numb_surrogates}_PGDi{pgd_iterations}_generator{args.w_idx}_TREMBA')


# run attacks
attack_evaluate()

