import torch
from torch.utils.data import DataLoader
from Attack.Baseline import Baseline
from Attack.NewAttack import Proposed
from Attack.Average0Q import Average0
from Attack.NewAttackUpdated_v2 import newProposed
from config import SURROGATE_NAMES, VICTIM_NAMES
from Utils.load_models import load_model, load_dataset, load_surrogates
from PIL import Image
from Utils.utils import save_json, generate_time, normalize
from Utils.load_models import load_dataset, load_model
import argparse

# we have to iterate over all the sample and save the logits of each model, both victim and surrogates
device = 'cuda:0'
batch_size = 1000

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
ens_surrogates = [load_model(surrogate, device=device).to(device) for surrogate in SURROGATE_NAMES]
victim_models = [load_model(victim_name, device=device).to(device) for victim_name in VICTIM_NAMES]

simlarity_dict_vict = {}
simlarity_dict_surr = {}

for i, victim in enumerate(victim_models):
    print(VICTIM_NAMES[i])
    simlarity_dict_vict[VICTIM_NAMES[i]] = []
    for idx in range(batch_size):
           for idx in range(batch_size):
            image = images[idx]
            label = labels[idx]
            target = targets[idx]

            victim.eval()
            logits = victim(image)
            logits = logits.detach()
            pred_label = logits.argmax()
            simlarity_dict_vict[VICTIM_NAMES[i]].append({f'{idx}': {'true_label': label.item(),
                                                            'pred_label': pred_label.item(),
                                                            'logits': logits.cpu().numpy().tolist()}})

save_json(simlarity_dict_vict, f'{generate_time()}_Victs_sim_eval_b{batch_size}_')

for i, surr in enumerate(ens_surrogates):
    print(SURROGATE_NAMES[i])
    simlarity_dict_surr[SURROGATE_NAMES[i]] = []
    for idx in range(batch_size):
           for idx in range(batch_size):
            image = images[idx]
            label = labels[idx]
            target = targets[idx]

            surr.eval()
            logits = surr(image)
            logits = logits.detach()
            pred_label = logits.argmax()
            simlarity_dict_surr[SURROGATE_NAMES[i]].append({f'{idx}': {'true_label': label.item(),
                                                            'pred_label': pred_label.item(),
                                                            'logits': logits.cpu().numpy().tolist()}})

save_json(simlarity_dict_surr, f'{generate_time()}_surrEns_sim_eval_b{batch_size}_')


