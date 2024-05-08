import torch
torch.manual_seed(0)
from torch.utils.data import DataLoader
from Attack.Baseline import Baseline
from Attack.NewAttack import Proposed
from Attack.Average0Q import Average0
from config import SURROGATE_NAMES, VICTIM_NAMES
from Utils.load_models import load_model, load_dataset, load_surrogates
from PIL import Image
from Utils.utils import save_json, generate_time, normalize
from Utils.load_models import load_dataset, load_model
import argparse

parser = argparse.ArgumentParser(description="Run Attacks")
parser.add_argument('--attack_type', type=str, default='B', choices=['B', 'A'], help='Type of attack')
parser.add_argument('--victim', type=str, default='vgg19',
                    choices=['resnext50_32x4d', 'vgg19', 'densenet121', 'alexnet', 'swin_s', 'shufflenet_v2_x2_0',
                             'regnet_y_32gf', 'efficientnet_v2_l'], help='Type of attack')
parser.add_argument('--n_surrogates', type=int, default=20, help='Number of Surrogates')
parser.add_argument('--batch_size', type=int, default=10, help='Number of sample to evaluate')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda:0', 'cuda:1', 'cuda:2', 'cpu'], help='Device to use (cpu, cuda:0, '
                                                                                        'cuda:1)')
parser.add_argument('--attack_iterations', type=int, default=40, help='Number of attack iterations')
parser.add_argument('--pgd_iterations', type=int, default=10, help='Number of pgd iterations')
parser.add_argument('--loss', type=str, default='CW', choices=['CW', 'CE'], help='Loss function')
parser.add_argument('--pool', type=str, default='0', choices=['0', '1', '2'], help='Pool of surrogates')
parser.add_argument('--eps', type=str, default='0', help='Perturbation Size, 0 16/255, 1 8/255 2 4/255')
parser.add_argument('--mul', type=int, default=1, help='multiplier step')
args = parser.parse_args()

multiplier = int(args.mul)

eps_dict = {'0': [16/255, 1], 
            '1':[8/255, 1], 
            '2':[4/255, 2]}
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
alpha = tm[1]* multiplier * 3 * eps / 10
x = alpha
pool = int(args.pool)

attack_dict = {'B': Baseline, 'A': Average0}
attack = attack_dict[attack_type]

if pool == 0:
    surrogates = [surr for surr in SURROGATE_NAMES[:numb_surrogates]]
elif pool == 1:

    surrogates_pool = ["densenet161", "efficientnet_v2_l", "regnet_y_16gf",
                  "resnet101", 'inception_v3', 'mnasnet1_0', 'googlenet',
                  'resnet18', 'convnext_small', 'mobilenet_v3_small']

    surrogates = [surr for surr in surrogates_pool[:numb_surrogates]]

elif pool == 2:
    surrogates_pool =['inception_v3' ,'mobilenet_v3_small', 'squeezenet1_1', 'googlenet', 'resnet18',
                  'mnasnet1_0', 'densenet161', 'efficientnet_b0',
                  'regnet_y_400mf', 'resnext101_32x8d']
    surrogates = [surr for surr in surrogates_pool[:numb_surrogates]]

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
ens_surrogates = [load_model(surrogate, device=device).to(device) for surrogate in surrogates]
victim_model = load_model(victim_name, device=device).to(device)



def attack_evaluate():
    global attack, attack_id, surrogates
    global numb_surrogates, eps, pgd_iterations, lr_w, attack_iterations, alpha, x, batch_size, loss
    global victim_model, ens_surrogates, victim_name
    global images, labels, targets

    results_dict = {}
    print(eps)

    results_dict['ensemble'] = surrogates
    print("Surrogates", len(ens_surrogates))
    # instantiate attack
    attacker = attack(victim_model, ens_surrogates, attack_iterations,
                      alpha, lr_w, eps, pgd_iterations, loss=loss, device=device)

    for idx in range(batch_size):
        image = images[idx]
        label = labels[idx]
        target = targets[idx]

        print(f"\n-------- Sample Number:{idx} - victim {victim_name} -------- ")
        print(f"### {str(attack_dict[attack_type].__name__)} ###")
        query_b, loss_list_b, n_iter_b, weights_b = attacker.forward(image, label, target)
        results_dict[f'{idx}'] = {'query': query_b,
                                  'loss_list': loss_list_b,
                                  'n_iter': n_iter_b,
                                  'weights': weights_b}

    save_json(results_dict,
              f'{generate_time()}_{str(attack_dict[attack_type].__name__)}_{victim_name}_b{batch_size}_eps{str(eps)[0:5]}_alp{str(alpha)[0:5]}_pool{pool}_surr{numb_surrogates}_PGDi{pgd_iterations}')

# run attacks
attack_evaluate()
