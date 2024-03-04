import torch
from torchvision import transforms
from Utils.load_models import load_model, load_imagenet_1000, load_surrogates
from Utils.compute_alignment import compute_alignment
from Utils.CW_loss import CWLoss
from Utils.utils import save_json, generate_time
from config import DATASET_PATH, SURROGATE_NAMES
from PIL import Image
import argparse

parser = argparse.ArgumentParser(description="Run Alignment Computation")

parser.add_argument('--n_surrogates', type=int, default=20, help='Number of Surrogates')
parser.add_argument('--batch_size', type=int, default=10, help='Number of sample to evaluate')
parser.add_argument('--device', type=str, default='cpu', choices=['cuda', 'cpu'], help='Device to use (cpu, cuda:0, '
                                                                                       'cuda:1)')

args = parser.parse_args()
print("running alignment computation ")
# parameters
numb_surrogates = int(args.n_surrogates)
device = args.device
batch_size = int(args.batch_size)
loss_fn = CWLoss()

# loading dataset
img_paths, true_labels, target_labels = load_imagenet_1000(dataset_root=DATASET_PATH)
samples_idx = [i for i in range(batch_size)]
# load models
victim_models = ['resnext50_32x4d', 'vgg19', 'densenet121']
ens_surrogates = load_surrogates(SURROGATE_NAMES[:numb_surrogates], device)

vic_dict = {}
to_tensor = transforms.ToTensor()
for victim in victim_models:
    victim_model = load_model(victim, device)
    alignment_samples = []
    for index in samples_idx:
        input = to_tensor(Image.open(img_paths[index]).convert('RGB')).to(device)
        label = torch.tensor(true_labels[index]).to(device)
        exp_name = f"idx{index}_f{true_labels[index]}_t{target_labels[index]}"
        target_label = target_labels[index]
        target = torch.tensor([target_label]).to(device)
        alignment_per_sample = []
        print(f"\n-------- Sample Number:{index} -------- ")
        alignment_dict = compute_alignment(input, victim_model, ens_surrogates, loss_fn, target)
        alignment_samples.append(alignment_dict)
    vic_dict[victim] = alignment_samples

    save_json(vic_dict, f'{generate_time()}_batch_{batch_size}_n_surr_{numb_surrogates}_alignment_exp', "Results")
