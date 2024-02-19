import torch
from torchvision import transforms
from Attack.Baseline import Baseline
from Attack.NewAttack import Proposed
from config import DATASET_PATH, SURROGATE_NAMES
from Utils.load_models import load_model, load_imagenet_1000, load_surrogates
from PIL import Image
from Utils.utils import save_json, generate_time

# attacks parameters
numb_surrogates = 10
eps = 16
pgd_iterations = 10
lr_w = 5e-3
attack_iterations = 40
alpha = 3 * eps / pgd_iterations
x = alpha
device = f'cpu'
batch_size = 5
loss = 'CW'

# loading dataset
img_paths, true_labels, target_labels = load_imagenet_1000(dataset_root=DATASET_PATH)
print(img_paths)
samples_idx = [i for i in range(batch_size)]
# load models
victim_models = ['resnext50_32x4d', 'vgg19', 'densenet121']
ens_surrogates = load_surrogates(SURROGATE_NAMES[:numb_surrogates], device)
victim_model = load_model(victim_models[1], device)

baseline_results_dict = {}
proposed_results_dict = {}

baseline_results_dict['ensemble'] = SURROGATE_NAMES[:numb_surrogates]
proposed_results_dict['ensemble'] = SURROGATE_NAMES[:numb_surrogates]

to_tensor = transforms.ToTensor()

# instantiate attacks
baseline = Baseline(victim_model, ens_surrogates, attack_iterations,
                    alpha, lr_w, eps, pgd_iterations, loss='CW', device='cpu')
proposed = Proposed(victim_model, ens_surrogates, attack_iterations,
                    alpha, lr_w, eps, pgd_iterations, loss='CW', device='cpu')

for index in samples_idx:
    image = to_tensor(Image.open(img_paths[index]).convert('RGB')).to(device)
    true_label = torch.tensor(true_labels[index]).to(device)
    target_label = torch.tensor([target_labels[index]]).to(device)

    print(f"\n-------- Sample Number:{index} -------- ")
    print("### Baseline ###")
    query_b, loss_list_b, logits_dist_b, n_iter_b, weights_b = baseline.forward(image, true_label, target_label)
    baseline_results_dict[f'{index}'] = {'query': query_b,
                                         'loss_list': loss_list_b,
                                         'logits_dist': logits_dist_b,
                                         'n_iter': n_iter_b,
                                         'weights': weights_b}

    print("\n")
    print("### Proposed ###")
    query_p, loss_list_p, logits_dist_p, n_iter_p, weights_p = proposed.forward(image, true_label, target_label)
    proposed_results_dict[f'{index}'] = {'query': query_p,
                                         'loss_list': loss_list_p,
                                         'logits_dist': logits_dist_p,
                                         'n_iter': n_iter_p,
                                         'weights': weights_p}

save_json(baseline_results_dict,
          f'{generate_time()}_BASELINE_victim_{victim_models[1]}_batch_{batch_size}_numb_surr{numb_surrogates}')
save_json(proposed_results_dict,
          f'{generate_time()}_PROPOSED_victim_{victim_models[1]}_batch_{batch_size}_numb_surr{numb_surrogates}')