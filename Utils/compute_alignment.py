import torch
from Utils.utils import normalize
from config import SURROGATE_NAMES


def compute_alignment(input, victim_model, ens_surrogates, loss_fn, target):
    alignment_dict = {}
    input = input.unsqueeze(dim=0)
    victim_model.eval()
    input.requires_grad_()
    input_tensor = normalize(input / 255)
    outputs = victim_model(input_tensor)
    lossv = loss_fn(outputs, target)
    lossv.backward(retain_graph=True)
    with torch.no_grad():
        vic_grad = input.grad.detach()

    for i, model in enumerate(ens_surrogates):
        input.grad.data.zero_()
        model.eval()
        outputs = model(input_tensor)
        loss = loss_fn(outputs, target)
        loss.backward(retain_graph=True)
        with torch.no_grad():
            surr_grad = input.grad.detach()
            g1 = torch.flatten(vic_grad)
            g2 = torch.flatten(surr_grad)
            alignment = torch.dot(g1, g2) / torch.norm(g1, p=2) * torch.norm(g2, p=2)
            alignment_dict[SURROGATE_NAMES[i]] = alignment.item()

    return alignment_dict
