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
parser.add_argument('--n_surrogates', type=int, default=20, help='Number of Surrogates')
parser.add_argument('--batch_size', type=int, default=10, help='Number of sample to evaluate')
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to use (cpu, cuda:0, '
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
eps = 16/255
lr_w = 5e-3
alpha = 3 * eps / pgd_iterations
x = alpha


def attack_evaluate():
    global numb_surrogates, eps, pgd_iterations, lr_w, attack_iterations, alpha, x, batch_size, loss
    global victim_model, ens_surrogates
    global images, labels, targets

    baseline_results_dict = {}
    proposed_results_dict = {}
    proposed1Q_results_dict = {}

    baseline_results_dict['ensemble'] = SURROGATE_NAMES[:numb_surrogates]
    proposed_results_dict['ensemble'] = SURROGATE_NAMES[:numb_surrogates]
    proposed1Q_results_dict['ensemble'] = SURROGATE_NAMES[:numb_surrogates]

    # instantiate attacks
    baseline = Baseline(victim_model, ens_surrogates, attack_iterations,
                        alpha, lr_w, eps, pgd_iterations, loss=loss, device=device)
    proposed = Proposed(victim_model, ens_surrogates, attack_iterations,
                        alpha, lr_w, eps, pgd_iterations, loss=loss, device=device)
    proposed1Q = Proposed1Q(victim_model, ens_surrogates, attack_iterations,
                        alpha, lr_w, eps, pgd_iterations, loss=loss, device=device)                    

    for idx in range(batch_size):
        image = images[idx]
        label = labels[idx]
        target = targets[idx]

        print(f"\n-------- Sample Number:{idx} -------- ")
        print("### Baseline ###")
        query_b, loss_list_b, logits_dist_b, n_iter_b, weights_b = baseline.forward(image, label, target)
        baseline_results_dict[f'{idx}'] = {'query': query_b,
                                        'loss_list': loss_list_b,
                                        'logits_dist': logits_dist_b,
                                        'n_iter': n_iter_b,
                                        'weights': weights_b}

        print("\n")
        print("### Proposed ###")
        query_p, loss_list_p, logits_dist_p, n_iter_p, weights_p = proposed.forward(image, label, target)
        proposed_results_dict[f'{idx}'] = {'query': query_p,
                                        'loss_list': loss_list_p,
                                        'logits_dist': logits_dist_p,
                                        'n_iter': n_iter_p,
                                        'weights': weights_p}
        print("### Proposed1Q ###")
        query_p, loss_list_p, logits_dist_p, n_iter_p, weights_p = proposed1Q.forward(image, label, target)
        proposed1Q_results_dict[f'{idx}'] = {'query': query_p,
                                        'loss_list': loss_list_p,
                                        'logits_dist': logits_dist_p,
                                        'n_iter': n_iter_p,
                                        'weights': weights_p}

    save_json(baseline_results_dict,
            f'{generate_time()}_BASELINE_victim_{VICTIM_NAMES[0]}_batch_{batch_size}_numb_surr{numb_surrogates}')
    save_json(proposed_results_dict,
            f'{generate_time()}_PROPOSED_victim_{VICTIM_NAMES[0]}_batch_{batch_size}_numb_surr{numb_surrogates}')
    save_json(proposed1Q_results_dict,
            f'{generate_time()}_PROPOSED_victim_{VICTIM_NAMES[0]}_batch_{batch_size}_numb_surr{numb_surrogates}')


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
victim_models = [load_model(victim, device=device).to(device) for victim in VICTIM_NAMES]
ens_surrogates = [load_model(surrogate, device=device).to(device) for surrogate in SURROGATE_NAMES]
victim_model = victim_models[0]

# run attacks
attack_evaluate()
