import torch
from torchvision import transforms
from Utils.load_models import load_model, load_imagenet_1000, load_surrogates
from Utils.compute_alignment import compute_alignment
from Utils.CW_loss import CWLoss
from Utils.utils import save_json, generate_time
from config import DATASET_PATH, SURROGATE_NAMES
from PIL import Image

# parameters
numb_surrogates = 20
device = f'cpu'
batch_size = 10
loss_fn = CWLoss()

# loading dataset
img_paths, true_labels, target_labels = load_imagenet_1000(dataset_root=DATASET_PATH)
print(img_paths)
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

save_json(vic_dict, f'{generate_time()}_batch_{batch_size}_alignment_exp')
