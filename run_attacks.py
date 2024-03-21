import torch
from torch.utils.data import DataLoader
from Attack.Baseline import Baseline
from Attack.NewAttack import Proposed
from Attack.NewAttack1Q import Proposed1Q
from config import SURROGATE_NAMES, VICTIM_NAMES
from Utils.load_models import load_model, load_dataset, load_surrogates
from PIL import Image
from Utils.utils import save_json, generate_time, normalize
from Utils.load_models import load_dataset, load_model
import argparse

parser = argparse.ArgumentParser(description="Run Attacks")
parser.add_argument('--attack_type', type=str, default='B', choices=['B', 'P', 'P1Q'], help='Type of attack')
parser.add_argument('--victim', type=str, default='vgg19', choices=['resnext50_32x4d', 'vgg19','densenet121'], help='Type of attack')
parser.add_argument('--n_surrogates', type=int, default=20, help='Number of Surrogates')
parser.add_argument('--batch_size', type=int, default=10, help='Number of sample to evaluate')
parser.add_argument('--device', type=str, default='cuda', choices=['cuda', 'cpu'], help='Device to use (cpu, cuda:0, '
                                                                                       'cuda:1)')
parser.add_argument('--attack_iterations', type=int, default=40, help='Number of attack iterations')
parser.add_argument('--pgd_iterations', type=int, default=10, help='Number of pgd iterations')
parser.add_argument('--loss', type=str, default='CW', choices=['CW', 'CE'], help='Loss function')

args = parser.parse_args()

# attacks parameters
numb_surrogates = int(args.n_surrogates)
batch_size = int(args.batch_size)
device = args.device
loss = args.loss
attack_iterations = int(args.attack_iterations)
pgd_iterations = int(args.pgd_iterations)
attack_type = args.attack_type
victim_name = args.victim
eps = 16/255
lr_w = 5e-3
alpha = 3 * eps / pgd_iterations
x = alpha

attack_dict = {'B': Baseline, 'P':Proposed, 'P1Q': Proposed1Q}
attack = attack_dict[attack_type]


def attack_evaluate():
    global attack, attack_id    
    global numb_surrogates, eps, pgd_iterations, lr_w, attack_iterations, alpha, x, batch_size, loss
    global victim_model, ens_surrogates, victim_name
    global images, labels, targets

    results_dict = {}


    results_dict['ensemble'] = SURROGATE_NAMES[:numb_surrogates]

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
            f'{generate_time()}_{str(attack_dict[attack_type].__name__)}_victim_{victim_name}_batch_{batch_size}_numb_surr{numb_surrogates}')

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
#victim_models = [load_model(victim, device=device).to(device) for victim in VICTIM_NAMES]
ens_surrogates = [load_model(surrogate, device=device).to(device) for surrogate in SURROGATE_NAMES]

victim_model = load_model(victim_name, device=device).to(device)


# run attacks
attack_evaluate()
